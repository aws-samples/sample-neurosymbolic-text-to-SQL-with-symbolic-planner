"""Unit tests for schema-driven SMT typing in the BIRD runner.

When the equivalence checker compares the generated DRC against the
gold DRC, both sides reference the same predicate symbols (the
table names) but each side's column-type inference is local to its
own DRC. If the same column appears in both DRCs but with different
inferred sorts, the merged equivalence script declares the predicate
once with one signature and the two sides emit constants with
mismatched sorts; cvc5 rejects the script before solving.

The runner now derives a ``{column_name: "String"}`` map from
``test_case.schema`` and threads it through to the equivalence
checker so cvc5 sees one consistent column sort across both sides.

These tests cover:

1. The pure-function helper ``_schema_types_from_test_case``:
   - Recognises common SQLite string types (TEXT, VARCHAR, CHAR).
   - Treats numeric / date / blob types as Int (cvc5's default sort).
   - Empty / unparseable / blank schemas yield an empty dict (the
     equivalence checker then falls back to per-DRC inference,
     which is the pre-fix behaviour).
   - When the same column appears with different declared types
     across multiple tables, the union takes ``"String"`` so a
     real string column can never be under-declared.
2. End-to-end through :func:`bird_benchmark.runner.run_one`:
   - The schema_types map is computed and passed to the equivalence
     checker.
   - The projection-tolerance retry (Option 2 from earlier) also
     receives the same map so the second call cannot disagree
     about column sorts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bird_benchmark.runner import _schema_types_from_test_case, run_one
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
# Pure-function: _schema_types_from_test_case
# ---------------------------------------------------------------------------


def _make_case(schema: str) -> TestCase:
    return TestCase(
        test_case_id="dev_test",
        split="dev",
        db_id="t",
        schema=schema,
        question="?",
        evidence="",
        gold_sql="SELECT 1",
    )


def test_schema_types_recognises_text_and_varchar_columns():
    """SQLite TEXT / VARCHAR / CHAR columns map to ``"String"``."""
    case = _make_case(
        "CREATE TABLE transactions_1k ("
        "TransactionID INT, Date TEXT, Time VARCHAR(8), "
        "Note CHAR(64), Amount INT);"
    )
    types = _schema_types_from_test_case(case)
    assert types == {"Date": "String", "Time": "String", "Note": "String"}


def test_schema_types_excludes_int_and_real_columns():
    """Numeric / blob / date types stay out of the dict (default Int sort)."""
    case = _make_case(
        "CREATE TABLE t (id INT, x REAL, y NUMERIC, z BLOB, d DATE);"
    )
    # The equivalence checker only needs the String overrides; everything
    # else defaults to Int.
    types = _schema_types_from_test_case(case)
    # ``DATE`` is treated as Int by the planner's typing rule (the
    # planner uses CURRENT_DATE / DATE_SUB integer arithmetic for
    # date math), so the dict is empty.
    assert types == {}


def test_schema_types_unions_string_across_tables():
    """A column declared TEXT in one table and INT in another resolves to String."""
    case = _make_case(
        "CREATE TABLE a (shared TEXT);\n"
        "CREATE TABLE b (shared INT);"
    )
    types = _schema_types_from_test_case(case)
    # Conservative: if any declaration says String, the column is String.
    # Otherwise the equivalence checker would under-declare the column
    # and crash whichever side meant it as a literal string.
    assert types.get("shared") == "String"


def test_schema_types_empty_for_blank_schema():
    """An empty schema yields an empty dict; the checker falls back to per-DRC inference."""
    assert _schema_types_from_test_case(_make_case("")) == {}
    assert _schema_types_from_test_case(_make_case("   \n  ")) == {}


def test_schema_types_empty_for_unparseable_schema():
    """A garbled schema doesn't raise — yields an empty dict."""
    case = _make_case("not actually a CREATE TABLE statement at all")
    assert _schema_types_from_test_case(case) == {}


def test_schema_types_handles_quoted_identifiers():
    """The BIRD distribution sometimes quotes identifiers; the parser handles it."""
    case = _make_case(
        'CREATE TABLE "transactions_1k" ('
        "TransactionID INTEGER primary key autoincrement,"
        "Date          DATE,"
        "Time          TEXT,"
        "Price         REAL"
        ");"
    )
    types = _schema_types_from_test_case(case)
    # ``Time`` is the only declared TEXT column in this fixture.
    assert types == {"Time": "String"}


# ---------------------------------------------------------------------------
# End-to-end: run_one threads schema_types through
# ---------------------------------------------------------------------------


def _bird_test_case() -> TestCase:
    """Reproduce the dev_1519 schema shape: TEXT columns next to INTs."""
    return TestCase(
        test_case_id="dev_1519",
        split="dev",
        db_id="debit_card_specializing",
        schema=(
            'CREATE TABLE "transactions_1k" ('
            "TransactionID INTEGER primary key autoincrement,"
            "Date DATE,"
            "Time TEXT,"
            "GasStationID INTEGER,"
            "ProductID INTEGER,"
            "Price REAL"
            ");"
        ),
        question="What was the product id of the transaction at 21:20:00?",
        evidence="",
        gold_sql=(
            "SELECT ProductID FROM transactions_1k WHERE Time = '21:20:00'"
        ),
    )


def _make_options() -> RunOptions:
    return RunOptions(
        bird_root=Path("/tmp/bird"),
        split="dev",
        cvc5_timeout_seconds=30,
        per_test_timeout_seconds=60,
    )


def _planner_returning(result):
    async def _run(**kwargs):
        return result

    return _run


def _converter_returning(result):
    def _convert(sql, schema):
        return result

    return _convert


def _make_drc() -> DRCExpression:
    return DRCExpression(
        result_variables=[ColumnVariable(name="ProductID")],
        condition=MembershipNode(
            variables=["t1", "d", "tm", "g", "ProductID", "p"],
            relation="transactions_1k",
        ),
    )


def _make_planner_success() -> TextToSQLSuccess:
    drc = _make_drc()
    tree = OperationTree(root=TableLeafNode(table_name="transactions_1k"))
    return TextToSQLSuccess(
        sql="SELECT ProductID FROM transactions_1k WHERE Time = '21:20:00'",
        operation_tree=tree,
        target_expression=drc,
        target_query=drc,
    )


@pytest.mark.asyncio
async def test_run_one_threads_schema_types_to_equivalence_checker():
    """The runner derives schema_types from the schema and passes it through."""

    captured: list[dict[str, str] | None] = []

    async def _capturing_check(expr1, expr2, config, *, schema_types=None, **kwargs):
        captured.append(schema_types)
        return EquivalentResult()

    rr = await run_one(
        _bird_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_capturing_check,
    )

    assert rr.underlying_verdict == Verdict.equivalent
    assert len(captured) == 1
    schema_types = captured[0]
    assert schema_types is not None
    # The fixture declares ``Time`` as TEXT, ``Date`` as DATE (treated
    # as Int), and the rest as numeric. So the schema_types map must
    # contain ``Time`` and only ``Time``.
    assert schema_types.get("Time") == "String"
    assert "Date" not in schema_types
    assert "GasStationID" not in schema_types


@pytest.mark.asyncio
async def test_run_one_passes_schema_types_to_projection_retry_too():
    """Both the original equivalence call and the prefix-tolerance retry get the same schema_types."""

    captured_kwargs: list[dict] = []

    async def _switching_check(expr1, expr2, config, *, schema_types=None, **kwargs):
        captured_kwargs.append({"schema_types": schema_types})
        # First call (full DRC vs gold) -> not_equivalent, second call
        # (truncated DRC) -> equivalent. This exercises the
        # projection-tolerance retry path with the schema_types map
        # already in flight.
        if len(captured_kwargs) == 1:
            return NotEquivalentResult()
        return EquivalentResult()

    # Generated has 2 result vars, gold has 1, prefix matches → retry fires.
    generated = DRCExpression(
        result_variables=[
            ColumnVariable(name="ProductID"),
            AggregateVariable(function="SUM", column="Price"),
        ],
        condition=MembershipNode(
            variables=["t1", "d", "tm", "g", "ProductID", "p"],
            relation="transactions_1k",
        ),
    )
    gold = DRCExpression(
        result_variables=[ColumnVariable(name="ProductID")],
        condition=MembershipNode(
            variables=["t1", "d", "tm", "g", "ProductID", "p"],
            relation="transactions_1k",
        ),
    )

    success = TextToSQLSuccess(
        sql="SELECT ProductID, SUM(Price) FROM transactions_1k GROUP BY ProductID",
        operation_tree=OperationTree(root=TableLeafNode(table_name="transactions_1k")),
        target_expression=generated,
        target_query=generated,
    )

    rr = await run_one(
        _bird_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(success),
        convert_sql_callable=_converter_returning(gold),
        check_equivalence_callable=_switching_check,
    )

    # Both calls happened.
    assert len(captured_kwargs) == 2
    # Both calls received the SAME schema_types map. If only the first
    # received it, the retry would reintroduce the sort mismatch the
    # whole exercise was designed to prevent.
    assert captured_kwargs[0]["schema_types"] == captured_kwargs[1]["schema_types"]
    assert captured_kwargs[0]["schema_types"].get("Time") == "String"
    # And the retry rescue still works.
    assert rr.underlying_verdict == Verdict.equivalent
