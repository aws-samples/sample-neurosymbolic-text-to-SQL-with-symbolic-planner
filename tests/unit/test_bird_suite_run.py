"""Smoke tests for ``bird_benchmark.runner.run_suite`` (task 12.1).

These tests exercise the wiring of the Suite_Run driver: per-Test_Case
iteration, manifest append, progress lines, the unhandled-exception →
``planner_failed`` mapping, the ``--resume`` skip, and the JSON +
markdown report fan-out at the end of the run. They do not invoke
Bedrock or cvc5; the planner / converter / equivalence / loader /
reporter callables are all replaced via the keyword-only injection
points on :func:`run_suite`.

Comprehensive property tests for full-suite coverage, resume
correctness, and progress-line / manifest correspondence live in tasks
12.2 / 12.3 / 12.4; this module's job is to confirm the suite-mode
driver hangs together end-to-end.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable

import pytest

from bird_benchmark.manifest import Manifest
from bird_benchmark.runner import run_suite
from bird_benchmark.types import (
    RunOptions,
    RunResult,
    SkippedTestCase,
    SuiteSummary,
    TestCase,
    Verdict,
)
from text_to_sql_planner.equivalence import EquivalentResult
from text_to_sql_planner.main import TextToSQLSuccess
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    MembershipNode,
)
from text_to_sql_planner.types.operation_tree import (
    OperationTree,
    TableLeafNode,
)


# --- Test fixtures ----------------------------------------------------


def _make_test_case(
    *,
    test_case_id: str = "dev_1",
    question: str = "How many people are there?",
) -> TestCase:
    return TestCase(
        test_case_id=test_case_id,
        split="dev",
        db_id="people",
        schema="CREATE TABLE people (id INT, name TEXT);",
        question=question,
        evidence="",
        gold_sql="SELECT COUNT(*) FROM people;",
    )


def _make_options(
    tmp_path: Path,
    *,
    resume: bool = False,
) -> RunOptions:
    """Build a ``RunOptions`` whose paths live under ``tmp_path``.

    The manifest and both report paths are placed in the per-test
    temporary directory so the suite can write real JSONL / JSON /
    markdown files without bleeding state between tests.
    """

    return RunOptions(
        bird_root=Path("/tmp/bird"),
        split="dev",
        cvc5_timeout_seconds=30,
        per_test_timeout_seconds=60,
        manifest_path=tmp_path / "manifest.jsonl",
        resume=resume,
        report_json_path=tmp_path / "report.json",
        report_md_path=tmp_path / "report.md",
    )


def _make_drc() -> DRCExpression:
    return DRCExpression(
        result_variables=[ColumnVariable(name="x")],
        condition=MembershipNode(variables=["x"], relation="people"),
    )


def _make_planner_success() -> TextToSQLSuccess:
    drc = _make_drc()
    tree = OperationTree(root=TableLeafNode(table_name="people"))
    return TextToSQLSuccess(
        sql="SELECT id FROM people;",
        operation_tree=tree,
        target_expression=drc,
        target_query=drc,
    )


class _StubLoader:
    """Minimal :class:`BirdLoader` substitute returning a fixed stream."""

    def __init__(self, items: Iterable[TestCase | SkippedTestCase]):
        self._items = list(items)

    def load(self):
        return iter(self._items)


def _loader_factory_returning(
    items: Iterable[TestCase | SkippedTestCase],
):
    materialised = list(items)

    def _factory(config):  # noqa: ARG001
        return _StubLoader(materialised)

    return _factory


def _planner_returning(result):
    async def _run(**kwargs):
        return result

    return _run


def _converter_returning(result):
    def _convert(sql, schema):
        return result

    return _convert


def _equivalence_returning(result):
    async def _check(*args, **kwargs):
        return result

    return _check


# --- Happy path -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_suite_happy_path_tallies_appends_and_prints(tmp_path):
    """Two equivalent Test_Cases: tally, manifest, progress, reports.

    Validates the load-bearing run_suite contract end-to-end:

    * every Test_Case gets exactly one Run_Result (Req 8.1);
    * the manifest contains one JSONL line per Test_Case (Req 8.2);
    * stdout has one progress line per Test_Case in the same order
      (Req 8.4);
    * the summary tally is correct;
    * the JSON and markdown reports are written to the configured
      paths (Req 9.1 / 9.2).
    """

    test_cases = [
        _make_test_case(test_case_id="dev_1"),
        _make_test_case(test_case_id="dev_2"),
    ]

    stdout = io.StringIO()
    stderr = io.StringIO()
    options = _make_options(tmp_path)

    summary = await run_suite(
        options,
        expected_fail=set(),
        stdout=stdout,
        stderr=stderr,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        loader_factory=_loader_factory_returning(test_cases),
    )

    # ``SuiteSummary`` shape ------------------------------------------
    assert isinstance(summary, SuiteSummary)
    assert summary.failed_to_write_report is False
    assert summary.stale_expected_fail_ids == set()
    assert len(summary.results) == 2
    assert summary.counts[Verdict.equivalent] == 2
    # Every Verdict appears in the counts dict (Req 9.5 mirrors this on
    # the JSON report; the in-memory tally has the same shape).
    for verdict in Verdict:
        assert verdict in summary.counts

    # Manifest one JSONL line per Test_Case (Req 8.2) ------------------
    manifest_lines = options.manifest_path.read_text().splitlines()
    assert len(manifest_lines) == 2
    parsed = [json.loads(line) for line in manifest_lines]
    assert [r["test_case_id"] for r in parsed] == ["dev_1", "dev_2"]
    assert all(r["reported_verdict"] == "equivalent" for r in parsed)

    # Progress lines mirror manifest entries (Req 8.4) -----------------
    progress = stdout.getvalue().splitlines()
    assert progress == ["dev_1 equivalent", "dev_2 equivalent"]

    # Reports written to configured paths (Req 9.1 / 9.2) --------------
    assert options.report_json_path.exists()
    assert options.report_md_path.exists()
    json_payload = json.loads(options.report_json_path.read_text())
    assert json_payload["summary"]["equivalent"] == 2
    assert len(json_payload["results"]) == 2


# --- Resume skip ------------------------------------------------------


@pytest.mark.asyncio
async def test_run_suite_resume_skips_completed_test_cases(tmp_path):
    """``--resume`` skips Test_Cases already in the manifest (Req 8.3).

    Pre-seeds the manifest with one completed Run_Result for ``dev_1``
    and asserts that:

    * the planner is called exactly once (for ``dev_2``);
    * the manifest grows by exactly one line;
    * the progress output lists only ``dev_2``.
    """

    options = _make_options(tmp_path, resume=True)

    # Seed the manifest with a completed Run_Result for ``dev_1`` so
    # the resume path has something to skip.
    options.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    seeded = RunResult(
        test_case_id="dev_1",
        underlying_verdict=Verdict.equivalent,
        reported_verdict=Verdict.equivalent,
        question="How many people are there?",
        evidence="",
        planner_input="How many people are there?",
        generated_sql="SELECT id FROM people;",
        gold_sql="SELECT COUNT(*) FROM people;",
        smt_script=None,
        reason="",
        error_code="",
    )
    with Manifest.open_for_append(options.manifest_path) as m:
        m.append(seeded)

    test_cases = [
        _make_test_case(test_case_id="dev_1"),
        _make_test_case(test_case_id="dev_2"),
    ]

    planner_calls: list[dict] = []

    async def _counting_planner(**kwargs):
        planner_calls.append(kwargs)
        return _make_planner_success()

    stdout = io.StringIO()
    stderr = io.StringIO()
    summary = await run_suite(
        options,
        expected_fail=set(),
        stdout=stdout,
        stderr=stderr,
        planner_callable=_counting_planner,
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        loader_factory=_loader_factory_returning(test_cases),
    )

    # Planner called exactly once (Req 8.3) ----------------------------
    assert len(planner_calls) == 1

    # The summary contains only the *new* Run_Result; the resumed entry
    # is already on disk so the suite driver does not re-tally it.
    assert len(summary.results) == 1
    assert summary.results[0].test_case_id == "dev_2"

    # Manifest now has two lines, both well-formed.
    manifest_lines = options.manifest_path.read_text().splitlines()
    assert len(manifest_lines) == 2
    parsed = [json.loads(line) for line in manifest_lines]
    assert [r["test_case_id"] for r in parsed] == ["dev_1", "dev_2"]

    # Progress output reflects only the work the suite actually did.
    progress = stdout.getvalue().splitlines()
    assert progress == ["dev_2 equivalent"]


# --- Unhandled-exception → planner_failed -----------------------------


@pytest.mark.asyncio
async def test_run_suite_unhandled_exception_becomes_planner_failed(tmp_path):
    """An unhandled exception from ``run_one`` -> Run_Result.planner_failed.

    Drives the planner to raise a synthetic ``RuntimeError`` and
    asserts that:

    * the suite continues past the failing Test_Case (Req 8.6);
    * the recorded Run_Result has ``reported_verdict == planner_failed``;
    * the ``reason`` string contains the exception type and message.

    The injection target is the planner callable. ``run_one`` already
    short-circuits planner exceptions to ``planner_failed`` internally,
    but the suite driver's catch-all is the safety net for any
    exception that escapes ``run_one`` (e.g. a future ``run_one``
    refactor that surfaces a non-planner exception). For this smoke
    test we use the planner-raises path because it is the easiest
    deterministic failure to inject without monkeypatching internal
    helpers; the resulting Run_Result still satisfies Req 8.6.
    """

    test_cases = [
        _make_test_case(test_case_id="dev_1"),
        _make_test_case(test_case_id="dev_2"),
    ]

    async def _raising_planner(**kwargs):
        # Match against the question to fail only the first Test_Case.
        if kwargs.get("question", "").startswith("How many people"):
            raise RuntimeError("synthetic planner crash")
        return _make_planner_success()

    # Different question on the second Test_Case so the planner can
    # distinguish them and only fail one. We rebuild the second case
    # in-place to keep the helper short.
    test_cases[1] = _make_test_case(
        test_case_id="dev_2", question="What is the average age?"
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    options = _make_options(tmp_path)

    summary = await run_suite(
        options,
        expected_fail=set(),
        stdout=stdout,
        stderr=stderr,
        planner_callable=_raising_planner,
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        loader_factory=_loader_factory_returning(test_cases),
    )

    # Both Test_Cases produced Run_Results -- the suite did not abort
    # on the first failure (Req 8.6).
    assert len(summary.results) == 2
    first, second = summary.results
    assert first.test_case_id == "dev_1"
    assert first.reported_verdict == Verdict.planner_failed
    # The reason names the exception type and message so an operator
    # can triage from the report alone.
    assert "RuntimeError" in first.reason
    assert "synthetic planner crash" in first.reason

    assert second.test_case_id == "dev_2"
    assert second.reported_verdict == Verdict.equivalent

    # Progress lines emitted for both Test_Cases (Req 8.4).
    progress = stdout.getvalue().splitlines()
    assert progress == ["dev_1 planner_failed", "dev_2 equivalent"]

    # Tally reflects the mixed verdicts.
    assert summary.counts[Verdict.planner_failed] == 1
    assert summary.counts[Verdict.equivalent] == 1
