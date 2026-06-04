"""Smoke tests for ``bird_benchmark.runner.run_single`` (task 10.1).

These tests exercise the wiring of the Single_Run driver: selector
validation, loader filtering, and JSON-to-stdout serialisation. They do
not invoke Bedrock or cvc5, and they do not require a real BIRD
download — the loader is replaced via the ``loader_factory`` injection
point and the planner / converter / equivalence callables are stubbed
the same way as in ``test_bird_runner.py``.

Comprehensive property tests for selector resolution and selector
errors live in tasks 10.2 / 10.3; this module's job is to confirm the
single-mode driver hangs together.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable

import pytest

from bird_benchmark.runner import (
    SingleSelectorError,
    run_single,
)
from bird_benchmark.types import (
    RunOptions,
    SingleSelector,
    SkippedTestCase,
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


def _make_options() -> RunOptions:
    return RunOptions(
        bird_root=Path("/tmp/bird"),
        split="dev",
        cvc5_timeout_seconds=30,
        per_test_timeout_seconds=60,
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
    """Minimal :class:`BirdLoader` substitute returning a fixed stream.

    ``run_single`` only calls ``.load()`` on the loader, so a duck-typed
    object is sufficient. Using a class instead of a closure makes the
    intent obvious in stack traces.
    """

    def __init__(self, items: Iterable[TestCase | SkippedTestCase]):
        self._items = list(items)

    def load(self):
        return iter(self._items)


def _loader_factory_returning(
    items: Iterable[TestCase | SkippedTestCase],
):
    """Build a ``loader_factory`` callable that ignores its config."""

    materialised = list(items)

    def _factory(config):  # noqa: ARG001 - factory ignores config in tests
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


# --- Selector validation ----------------------------------------------


@pytest.mark.asyncio
async def test_run_single_rejects_both_selectors_set():
    """Supplying both ``test_case_id`` and ``question_text`` is rejected (Req 7.6)."""
    selector = SingleSelector(test_case_id="dev_1", question_text="anything")
    with pytest.raises(SingleSelectorError) as excinfo:
        await run_single(
            selector,
            _make_options(),
            expected_fail=set(),
            stdout=io.StringIO(),
            loader_factory=_loader_factory_returning([]),
        )
    assert "exactly one" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_run_single_rejects_neither_selector_set():
    """Supplying neither selector is rejected (Req 7.6)."""
    selector = SingleSelector()
    with pytest.raises(SingleSelectorError) as excinfo:
        await run_single(
            selector,
            _make_options(),
            expected_fail=set(),
            stdout=io.StringIO(),
            loader_factory=_loader_factory_returning([]),
        )
    assert "exactly one" in str(excinfo.value).lower()


# --- Selector resolution ----------------------------------------------


@pytest.mark.asyncio
async def test_run_single_zero_match_raises_with_selector_and_split():
    """Zero matches: the error names the selector and the split (Req 7.4)."""
    selector = SingleSelector(test_case_id="dev_999")
    with pytest.raises(SingleSelectorError) as excinfo:
        await run_single(
            selector,
            _make_options(),
            expected_fail=set(),
            stdout=io.StringIO(),
            loader_factory=_loader_factory_returning(
                [_make_test_case(test_case_id="dev_1")]
            ),
        )
    message = str(excinfo.value)
    assert "dev_999" in message
    assert "dev" in message  # the split name


@pytest.mark.asyncio
async def test_run_single_ambiguous_question_raises_with_count():
    """A question that matches twice fails with the match count (Req 7.5)."""
    selector = SingleSelector(question_text="How many people are there?")
    with pytest.raises(SingleSelectorError) as excinfo:
        await run_single(
            selector,
            _make_options(),
            expected_fail=set(),
            stdout=io.StringIO(),
            loader_factory=_loader_factory_returning(
                [
                    _make_test_case(test_case_id="dev_1"),
                    _make_test_case(test_case_id="dev_2"),
                ]
            ),
        )
    message = str(excinfo.value)
    assert "2" in message  # match count
    assert "How many people" in message


# --- End-to-end success path ------------------------------------------


@pytest.mark.asyncio
async def test_run_single_unique_match_runs_and_prints_json():
    """A unique match runs ``run_one`` and writes a JSON line to stdout (Req 7.3)."""
    selector = SingleSelector(test_case_id="dev_1")
    stdout = io.StringIO()
    result = await run_single(
        selector,
        _make_options(),
        expected_fail=set(),
        stdout=stdout,
        loader_factory=_loader_factory_returning(
            [
                _make_test_case(test_case_id="dev_1"),
                _make_test_case(
                    test_case_id="dev_2",
                    question="A different question?",
                ),
            ]
        ),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
    )

    assert result.test_case_id == "dev_1"
    assert result.reported_verdict == Verdict.equivalent

    # The JSON serialisation must include the Test_Case_ID, generated SQL,
    # gold SQL, and Verdict per Req 7.3. SMT_Script is None for an
    # equivalent verdict and is therefore expected as JSON null.
    line = stdout.getvalue().strip()
    payload = json.loads(line)
    assert payload["test_case_id"] == "dev_1"
    assert payload["reported_verdict"] == "equivalent"
    assert payload["generated_sql"] == "SELECT id FROM people;"
    assert payload["gold_sql"] == "SELECT COUNT(*) FROM people;"
    assert "smt_script" in payload  # field is present (None for equivalent)


@pytest.mark.asyncio
async def test_run_single_skips_skipped_testcases_when_resolving():
    """``SkippedTestCase`` records are not eligible matches."""
    selector = SingleSelector(test_case_id="dev_1")
    stdout = io.StringIO()
    result = await run_single(
        selector,
        _make_options(),
        expected_fail=set(),
        stdout=stdout,
        loader_factory=_loader_factory_returning(
            [
                SkippedTestCase(
                    test_case_id="dev_record_0",
                    reason="missing field 'question'",
                ),
                _make_test_case(test_case_id="dev_1"),
            ]
        ),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
    )
    assert result.test_case_id == "dev_1"
