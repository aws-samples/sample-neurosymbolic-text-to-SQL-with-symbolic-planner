"""Unit tests for the BIRD projection-tolerance retry.

BIRD's gold SQL for ranking-style questions ("which gas station has
the highest revenue") often projects only the entity column while the
planner — answering the same question — projects both the entity and
the ranking column. The two queries pick the same row but report
different arities.

The runner now retries the equivalence check on a copy of the
generated DRC truncated to the gold's arity, when the generated's
result variables strictly extend the gold's prefix-by-prefix. These
tests cover:

1. The pure-function ``_truncate_to_gold_arity`` helper:
   - Returns ``None`` for equal-arity inputs.
   - Returns ``None`` when generated has *fewer* result variables than gold.
   - Returns ``None`` when the prefix doesn't match.
   - Returns a properly-truncated ``DRCExpression`` when the generated
     extends the gold, including the aggregate case.
2. End-to-end through :func:`bird_benchmark.runner.run_one`: a stubbed
   equivalence checker that says ``not_equivalent`` for the full DRC
   and ``equivalent`` for the truncated DRC produces a final verdict
   of ``Verdict.equivalent``. The retry must NOT fire when the prefix
   doesn't match.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bird_benchmark.runner import _truncate_to_gold_arity, run_one
from bird_benchmark.types import RunOptions, TestCase, Verdict
from text_to_sql_planner.equivalence import (
    EquivalentResult,
    NotEquivalentResult,
)
from text_to_sql_planner.main import TextToSQLSuccess
from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ColumnVariable,
    DRCExpression,
    MembershipNode,
)
from text_to_sql_planner.types.operation_tree import OperationTree, TableLeafNode


# ---------------------------------------------------------------------------
# Pure-function: _truncate_to_gold_arity
# ---------------------------------------------------------------------------


def _drc(*result_vars) -> DRCExpression:
    """Build a tiny DRC with the given result variables and a fixed body."""
    return DRCExpression(
        result_variables=list(result_vars),
        condition=MembershipNode(variables=["x"], relation="t"),
    )


def test_truncate_returns_none_for_equal_arity():
    """Equal arity is the normal equivalence path; no truncation."""
    gen = _drc(ColumnVariable(name="a"), ColumnVariable(name="b"))
    gold = _drc(ColumnVariable(name="a"), ColumnVariable(name="b"))
    assert _truncate_to_gold_arity(gen, gold) is None


def test_truncate_returns_none_when_generated_is_shorter():
    """Generated < gold can't be safely truncated — would invent columns."""
    gen = _drc(ColumnVariable(name="a"))
    gold = _drc(ColumnVariable(name="a"), ColumnVariable(name="b"))
    assert _truncate_to_gold_arity(gen, gold) is None


def test_truncate_returns_none_when_prefix_does_not_match():
    """Different leading variables → no safe truncation."""
    gen = _drc(
        ColumnVariable(name="b"),
        ColumnVariable(name="a"),
    )
    gold = _drc(ColumnVariable(name="a"))
    assert _truncate_to_gold_arity(gen, gold) is None


def test_truncate_drops_trailing_extras_for_column_prefix():
    """The BIRD ranking case: gen=(X, Y), gold=(X) → truncate to (X)."""
    gen = _drc(
        ColumnVariable(name="GasStationID"),
        AggregateVariable(function="SUM", column="Price"),
    )
    gold = _drc(ColumnVariable(name="GasStationID"))

    truncated = _truncate_to_gold_arity(gen, gold)
    assert truncated is not None
    assert len(truncated.result_variables) == 1
    assert truncated.result_variables[0].type == "column"
    assert truncated.result_variables[0].name == "GasStationID"
    # The condition must be preserved unchanged — only the projection
    # narrows.
    assert truncated.condition is gen.condition


def test_truncate_handles_aggregate_in_prefix():
    """Aggregate columns in the prefix match by (function, column)."""
    gen = _drc(
        AggregateVariable(function="COUNT", column="emp_id"),
        ColumnVariable(name="dept_id"),
    )
    gold = _drc(AggregateVariable(function="COUNT", column="emp_id"))
    truncated = _truncate_to_gold_arity(gen, gold)
    assert truncated is not None
    assert truncated.result_variables[0].type == "aggregate"
    assert truncated.result_variables[0].function == "COUNT"


def test_truncate_rejects_mismatched_aggregate_function():
    """SUM vs COUNT in the prefix must not be claimed as a match."""
    gen = _drc(
        AggregateVariable(function="SUM", column="Price"),
        ColumnVariable(name="GasStationID"),
    )
    gold = _drc(AggregateVariable(function="COUNT", column="Price"))
    assert _truncate_to_gold_arity(gen, gold) is None


# ---------------------------------------------------------------------------
# End-to-end: run_one with a switching equivalence stub
# ---------------------------------------------------------------------------


def _make_test_case() -> TestCase:
    return TestCase(
        test_case_id="dev_1527",
        split="dev",
        db_id="financial",
        schema="CREATE TABLE transactions_1k (TransactionID INT, GasStationID INT, Price REAL);",
        question="Which gas station has the highest amount of revenue?",
        evidence="",
        gold_sql=(
            "SELECT GasStationID FROM transactions_1k "
            "GROUP BY GasStationID ORDER BY SUM(Price) DESC LIMIT 1"
        ),
    )


def _make_options() -> RunOptions:
    return RunOptions(
        bird_root=Path("/tmp/bird"),
        split="dev",
        cvc5_timeout_seconds=30,
        per_test_timeout_seconds=60,
    )


def _make_planner_success(*result_vars) -> TextToSQLSuccess:
    """Planner success whose target DRC has the given result variables."""
    drc = DRCExpression(
        result_variables=list(result_vars),
        condition=MembershipNode(
            variables=["TransactionID", "GasStationID", "Price"],
            relation="transactions_1k",
        ),
    )
    tree = OperationTree(root=TableLeafNode(table_name="transactions_1k"))
    return TextToSQLSuccess(
        sql="SELECT GasStationID, SUM(Price) FROM transactions_1k "
        "GROUP BY GasStationID ORDER BY SUM(Price) DESC LIMIT 1",
        operation_tree=tree,
        target_expression=drc,
        target_query=drc,
    )


def _converter_returning(result):
    """Synchronous converter callable returning ``result``."""

    def _convert(sql, schema):
        return result

    return _convert


def _planner_returning(result):
    """Async planner callable returning ``result``."""

    async def _run(**kwargs):
        return result

    return _run


@pytest.mark.asyncio
async def test_run_one_promotes_not_equivalent_to_equivalent_when_prefix_extends():
    """The ranking-question failure mode is rescued by the retry.

    The stub equivalence checker says:
        - first call (full generated DRC vs gold)  -> not_equivalent
        - second call (truncated generated DRC)   -> equivalent

    The runner must surface ``Verdict.equivalent`` because the truncated
    generated DRC is equivalent to the gold DRC, which under BIRD's
    row-set evaluation contract is the same answer.
    """

    # Generated has TWO result variables; gold has ONE.
    generated_drc = DRCExpression(
        result_variables=[
            ColumnVariable(name="GasStationID"),
            AggregateVariable(function="SUM", column="Price"),
        ],
        condition=MembershipNode(
            variables=["TransactionID", "GasStationID", "Price"],
            relation="transactions_1k",
        ),
    )
    gold_drc = DRCExpression(
        result_variables=[ColumnVariable(name="GasStationID")],
        condition=MembershipNode(
            variables=["TransactionID", "GasStationID", "Price"],
            relation="transactions_1k",
        ),
    )

    call_arities: list[int] = []

    async def _switching_check(expr1, expr2, config, *args, **kwargs):
        # Track the arity of the first argument so the assertions
        # below can confirm the second call uses the truncated DRC.
        call_arities.append(len(expr1.result_variables))
        if len(expr1.result_variables) == len(expr2.result_variables):
            return EquivalentResult()
        return NotEquivalentResult()

    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(
            _make_planner_success(
                ColumnVariable(name="GasStationID"),
                AggregateVariable(function="SUM", column="Price"),
            )
        ),
        convert_sql_callable=_converter_returning(gold_drc),
        check_equivalence_callable=_switching_check,
    )

    # Both calls must have happened and the second must have used the
    # truncated 1-arity DRC.
    assert call_arities == [2, 1]
    assert rr.underlying_verdict == Verdict.equivalent
    assert rr.reported_verdict == Verdict.equivalent


@pytest.mark.asyncio
async def test_run_one_does_not_retry_when_prefix_does_not_match():
    """When the prefix doesn't match, no retry happens and the verdict stays not_equivalent."""

    # Generated and gold disagree on the first column → no safe truncation.
    gold_drc = DRCExpression(
        result_variables=[ColumnVariable(name="other_column")],
        condition=MembershipNode(variables=["x"], relation="t"),
    )

    call_count = {"value": 0}

    async def _counting_check(*args, **kwargs):
        call_count["value"] += 1
        return NotEquivalentResult()

    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(
            _make_planner_success(
                ColumnVariable(name="GasStationID"),
                AggregateVariable(function="SUM", column="Price"),
            )
        ),
        convert_sql_callable=_converter_returning(gold_drc),
        check_equivalence_callable=_counting_check,
    )

    # Exactly one equivalence call: the prefix didn't match so the
    # retry path was skipped.
    assert call_count["value"] == 1
    assert rr.underlying_verdict == Verdict.not_equivalent


@pytest.mark.asyncio
async def test_run_one_does_not_retry_when_arities_match():
    """Equal-arity inputs use the normal path; no retry call is made."""

    same_arity_gold = DRCExpression(
        result_variables=[ColumnVariable(name="GasStationID")],
        condition=MembershipNode(variables=["x"], relation="t"),
    )

    call_count = {"value": 0}

    async def _counting_check(*args, **kwargs):
        call_count["value"] += 1
        return NotEquivalentResult()

    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(
            _make_planner_success(ColumnVariable(name="GasStationID"))
        ),
        convert_sql_callable=_converter_returning(same_arity_gold),
        check_equivalence_callable=_counting_check,
    )

    # Equal arity means no retry path is taken.
    assert call_count["value"] == 1
    assert rr.underlying_verdict == Verdict.not_equivalent
