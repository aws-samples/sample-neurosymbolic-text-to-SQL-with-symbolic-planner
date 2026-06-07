"""Unit tests for the BIRD ``sample`` subcommand.

The sampler is the diagnostic verb between ``single`` (one Test_Case)
and ``suite`` (the whole split). It draws ``count`` Test_Cases at
random with a deterministic seed and reports both the cvc5 logical-
equivalence rate and the SQLite execution-equivalence rate, plus the
disagreement cells.

These tests cover:

1. End-to-end: stub planner / converter / equivalence checker /
   executor → the sampler runs the right number of Test_Cases,
   tallies counts correctly, identifies disagreement cells.
2. Determinism: same seed → same selection.
3. Edge cases: empty split, requested > eligible, validation
   errors on count / seed.
4. CLI plumbing: the new ``sample`` subparser parses, dispatches to
   :func:`run_sample`, and maps validation errors to
   ``EXIT_CONFIG_ERROR``.
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pytest

from bird_benchmark import cli
from bird_benchmark.cli import EXIT_CONFIG_ERROR, EXIT_OK
from bird_benchmark.runner import run_sample
from bird_benchmark.types import (
    ExecutionResult,
    ExecutionStatus,
    RunOptions,
    SampleSummary,
    SkippedTestCase,
    TestCase,
    Verdict,
)
from text_to_sql_planner.equivalence import (
    EquivalentResult,
    NotEquivalentResult,
)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_drc(name: str = "x", relation: str = "T") -> DRCExpression:
    return DRCExpression(
        result_variables=[ColumnVariable(name=name)],
        condition=MembershipNode(variables=[name], relation=relation),
    )


def _make_planner_success(sql: str = "SELECT id FROM t") -> TextToSQLSuccess:
    drc = _make_drc()
    tree = OperationTree(root=TableLeafNode(table_name="t"))
    return TextToSQLSuccess(
        sql=sql,
        operation_tree=tree,
        target_expression=drc,
        target_query=drc,
    )


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


def _exec_returning(result: ExecutionResult):
    def _exec(**kwargs):
        return result

    return _exec


def _make_test_case(idx: int) -> TestCase:
    return TestCase(
        test_case_id=f"dev_{idx}",
        split="dev",
        db_id="t",
        schema="CREATE TABLE t (id INT)",
        question=f"q{idx}",
        evidence="",
        gold_sql="SELECT id FROM t",
    )


def _make_loader(items: list[Any]):
    """Return a loader_factory that yields ``items`` in order."""

    class _Loader:
        def __init__(self, *_args, **_kwargs):
            pass

        def load(self) -> Iterable[Any]:
            yield from items

    def _factory(_config):
        return _Loader()

    return _factory


def _make_options(tmp_path: Path) -> RunOptions:
    return RunOptions(
        bird_root=tmp_path / "bird",
        split="dev",
        cvc5_timeout_seconds=30,
        per_test_timeout_seconds=60,
    )


# ---------------------------------------------------------------------------
# Determinism: same seed → same selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sample_with_same_seed_picks_same_test_cases(tmp_path):
    """Two runs with identical seed + count + split produce identical samples."""

    cases = [_make_test_case(i) for i in range(20)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    out = io.StringIO()

    summary_a = await run_sample(
        options,
        count=5,
        seed=42,
        expected_fail=set(),
        stdout=out,
        loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(
                status=ExecutionStatus.match,
                set_match=True,
                multiset_match=True,
            )
        ),
    )

    summary_b = await run_sample(
        options,
        count=5,
        seed=42,
        expected_fail=set(),
        stdout=io.StringIO(),
        loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(
                status=ExecutionStatus.match,
                set_match=True,
                multiset_match=True,
            )
        ),
    )

    ids_a = [r.test_case_id for r in summary_a.results]
    ids_b = [r.test_case_id for r in summary_b.results]
    assert ids_a == ids_b


@pytest.mark.asyncio
async def test_run_sample_with_different_seeds_picks_different_test_cases(tmp_path):
    """Different seeds produce different selections (sanity: not always equal)."""

    cases = [_make_test_case(i) for i in range(50)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)

    a = await run_sample(
        options, count=5, seed=1,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )
    b = await run_sample(
        options, count=5, seed=999,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    ids_a = [r.test_case_id for r in a.results]
    ids_b = [r.test_case_id for r in b.results]
    # Both runs must produce 5 results; in the 50-case pool the
    # probability of identical selections under different seeds is
    # negligible.
    assert ids_a != ids_b


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sample_records_logical_equivalence_rate(tmp_path):
    """Verdict counts and the equivalent-rate match the stub's behaviour."""

    cases = [_make_test_case(i) for i in range(10)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)

    summary = await run_sample(
        options, count=10, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        # Every case → cvc5 says equivalent.
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(
                status=ExecutionStatus.match,
                set_match=True, multiset_match=True,
            )
        ),
    )

    assert summary.sampled_count == 10
    assert summary.verdict_counts[Verdict.equivalent] == 10
    assert summary.execution_counts[ExecutionStatus.match] == 10
    assert summary.multiset_match_count == 10
    # No disagreements when both signals say "yes" everywhere.
    assert summary.logical_yes_exec_no == 0
    assert summary.logical_no_exec_yes == 0


@pytest.mark.asyncio
async def test_run_sample_counts_logical_no_exec_yes_disagreement(tmp_path):
    """The dev_1519 disagreement cell: cvc5 says no, exec says match.

    This is the highest-signal cell — it surfaces over-specified gold
    queries (extra joins, redundant filters) that don't change the
    answer on BIRD's snapshot.
    """

    cases = [_make_test_case(i) for i in range(8)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)

    summary = await run_sample(
        options, count=8, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        # Every case: cvc5 says NOT equivalent, exec says match.
        check_equivalence_callable=_equivalence_returning(NotEquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(
                status=ExecutionStatus.match,
                set_match=True, multiset_match=True,
            )
        ),
    )

    assert summary.verdict_counts[Verdict.not_equivalent] == 8
    assert summary.execution_counts[ExecutionStatus.match] == 8
    assert summary.logical_no_exec_yes == 8
    assert summary.logical_yes_exec_no == 0


@pytest.mark.asyncio
async def test_run_sample_counts_logical_yes_exec_no_disagreement(tmp_path):
    """The opposite cell: cvc5 says yes, exec says rows differ."""

    cases = [_make_test_case(i) for i in range(4)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)

    summary = await run_sample(
        options, count=4, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(
                status=ExecutionStatus.mismatch,
                set_match=False, multiset_match=False,
            )
        ),
    )
    assert summary.logical_yes_exec_no == 4


@pytest.mark.asyncio
async def test_run_sample_summary_emits_aggregate_lines_to_stdout(tmp_path):
    """The headline rates show up in stdout for the operator."""

    cases = [_make_test_case(i) for i in range(5)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    out = io.StringIO()

    await run_sample(
        options, count=5, seed=7,
        expected_fail=set(), stdout=out, loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(status=ExecutionStatus.match, set_match=True)
        ),
    )

    output = out.getvalue()
    assert "# Sample Summary" in output
    assert "Logically equivalent (cvc5):" in output
    assert "Execution-equivalent (set match):" in output
    assert "Disagreement (logical=yes, exec=no):" in output
    assert "Disagreement (logical=no, exec=yes):" in output
    # The seed should be surfaced so a re-run can reproduce.
    assert "seed=7" in output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sample_handles_count_larger_than_available(tmp_path):
    """When count > eligible, the sampler runs every eligible case once."""

    cases = [_make_test_case(i) for i in range(3)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)

    summary = await run_sample(
        options, count=100, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    assert summary.requested_count == 100
    assert summary.sampled_count == 3
    assert len(summary.results) == 3


@pytest.mark.asyncio
async def test_run_sample_skips_loader_skipped_records(tmp_path):
    """Loader-side skips don't enter the sampling pool."""

    items = [
        SkippedTestCase(test_case_id="dev_record_0", reason="missing question"),
        _make_test_case(1),
        SkippedTestCase(test_case_id="dev_record_2", reason="missing SQL"),
        _make_test_case(3),
    ]
    loader = _make_loader(items)
    options = _make_options(tmp_path)

    summary = await run_sample(
        options, count=2, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    # Only the two non-skipped Test_Cases are eligible; both are run.
    sampled_ids = {r.test_case_id for r in summary.results}
    assert sampled_ids == {"dev_1", "dev_3"}


@pytest.mark.asyncio
async def test_run_sample_handles_empty_split(tmp_path):
    """An empty split surfaces a non-fatal warning; sampled_count is 0."""

    loader = _make_loader([])
    options = _make_options(tmp_path)
    err = io.StringIO()

    summary = await run_sample(
        options, count=5, seed=0,
        expected_fail=set(),
        stdout=io.StringIO(),
        stderr=err,
        loader_factory=loader,
    )

    assert summary.sampled_count == 0
    assert summary.results == []
    assert "no eligible Test_Cases" in err.getvalue()


@pytest.mark.asyncio
async def test_run_sample_rejects_negative_count(tmp_path):
    """``count`` must be ≥ 1."""
    options = _make_options(tmp_path)
    with pytest.raises(ValueError):
        await run_sample(
            options, count=0, seed=0,
            expected_fail=set(),
            stdout=io.StringIO(),
            loader_factory=_make_loader([]),
        )


@pytest.mark.asyncio
async def test_run_sample_rejects_negative_seed(tmp_path):
    """``seed`` must be ≥ 0."""
    options = _make_options(tmp_path)
    with pytest.raises(ValueError):
        await run_sample(
            options, count=1, seed=-1,
            expected_fail=set(),
            stdout=io.StringIO(),
            loader_factory=_make_loader([]),
        )


# ---------------------------------------------------------------------------
# CLI: --count, --seed, dispatch
# ---------------------------------------------------------------------------


def test_cli_sample_subparser_requires_count_and_seed():
    """Argparse refuses missing required flags."""
    with pytest.raises(SystemExit):
        cli.main(
            ["sample", "--split", "dev", "--bird-root", "/tmp"],
            stderr=io.StringIO(), stdout=io.StringIO(),
        )


def test_cli_sample_options_carry_count_and_seed():
    """``--count`` / ``--seed`` parse and survive in the args namespace."""
    from bird_benchmark.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "sample",
            "--split", "dev",
            "--bird-root", "/tmp/bird",
            "--count", "25",
            "--seed", "42",
        ]
    )
    assert args.mode == "sample"
    assert args.count == 25
    assert args.seed == 42


def test_cli_sample_options_default_to_no_reports():
    """Sample mode defaults to no JSON / markdown report."""
    from bird_benchmark.cli import _build_parser, _options_from_args

    parser = _build_parser()
    args = parser.parse_args(
        [
            "sample",
            "--split", "dev", "--bird-root", "/tmp/bird",
            "--count", "1", "--seed", "0",
        ]
    )
    options = _options_from_args(args)
    assert options.report_json_path is None
    assert options.report_md_path is None


def test_cli_sample_invalid_count_maps_to_config_error(monkeypatch, tmp_path):
    """A ``ValueError`` from run_sample maps to ``EXIT_CONFIG_ERROR``."""
    # The argparse layer accepts any int, so a count of 0 reaches
    # run_sample, which raises ValueError. The CLI catches it and
    # surfaces ``EXIT_CONFIG_ERROR``.
    from bird_benchmark import cli as cli_mod

    async def _fake_sample(*_args, **_kwargs):
        raise ValueError("--count must be a positive integer, got 0")

    monkeypatch.setattr(cli_mod, "run_sample", _fake_sample)

    err = io.StringIO()
    code = cli.main(
        [
            "sample",
            "--split", "dev", "--bird-root", str(tmp_path),
            "--count", "0", "--seed", "0",
        ],
        stderr=err, stdout=io.StringIO(),
    )
    assert code == EXIT_CONFIG_ERROR
    assert "--count must be" in err.getvalue()


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sample_writes_json_report_when_path_given(tmp_path):
    """``options.report_json_path`` triggers a JSON report."""

    cases = [_make_test_case(i) for i in range(3)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    json_path = tmp_path / "sample.json"
    options.report_json_path = json_path

    await run_sample(
        options, count=3, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    assert json_path.is_file()
    payload = json.loads(json_path.read_text())
    assert "summary" in payload
    assert "results" in payload
    assert len(payload["results"]) == 3
