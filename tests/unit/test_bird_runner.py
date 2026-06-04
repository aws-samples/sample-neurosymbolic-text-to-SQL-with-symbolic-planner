"""Unit tests for ``bird_benchmark.runner.run_one``.

These tests exercise the per-Test_Case orchestration without invoking
Bedrock or cvc5: ``run_one`` accepts injected planner / converter /
equivalence callables specifically so the verdict-mapping logic can be
verified deterministically.

The properties under test mirror the design's verdict-mapping table and
the Expected_Fail override rules from task 9.1 (Requirements 3.1, 3.2,
3.3, 3.5, 5.1-5.9, 6.3, 6.4, 6.6, 11.4, 12.3, 12.5).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from bird_benchmark.runner import run_one
from bird_benchmark.types import (
    ConverterError,
    RunOptions,
    TestCase,
    Verdict,
)
from text_to_sql_planner.equivalence import (
    EquivalentResult,
    IndeterminateResult,
    NotEquivalentResult,
)
from text_to_sql_planner.main import TextToSQLFailure, TextToSQLSuccess
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    MembershipNode,
)
from text_to_sql_planner.types.errors import ErrorCode
from text_to_sql_planner.types.operation_tree import OperationTree, TableLeafNode


# --- Test fixtures ----------------------------------------------------


def _make_test_case(*, evidence: str = "", test_case_id: str = "dev_1") -> TestCase:
    return TestCase(
        test_case_id=test_case_id,
        split="dev",
        db_id="people",
        schema="CREATE TABLE people (id INT, name TEXT);",
        question="How many people are there?",
        evidence=evidence,
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
    """A trivial DRC expression: ``{x | people(x)}``."""
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


def _planner_returning(result):
    """Build a planner callable that always returns ``result``."""

    async def _run(**kwargs):
        return result

    return _run


def _converter_returning(result):
    """Build a converter callable that always returns ``result``."""

    def _convert(sql, schema):
        return result

    return _convert


def _equivalence_returning(result):
    """Build an equivalence callable that always returns ``result``."""

    async def _check(*args, **kwargs):
        return result

    return _check


# --- Verdict mapping --------------------------------------------------


@pytest.mark.asyncio
async def test_equivalent_result_maps_to_equivalent_verdict():
    """``EquivalentResult`` -> ``Verdict.equivalent``, no SMT script."""
    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
    )
    assert rr.underlying_verdict == Verdict.equivalent
    assert rr.reported_verdict == Verdict.equivalent
    assert rr.smt_script is None


@pytest.mark.asyncio
async def test_not_equivalent_result_maps_to_not_equivalent_with_smt():
    """``NotEquivalentResult`` captures an SMT script."""
    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(NotEquivalentResult()),
    )
    assert rr.underlying_verdict == Verdict.not_equivalent
    assert rr.smt_script is not None
    assert "people" in rr.smt_script  # the SMT rendering should include the relation


@pytest.mark.asyncio
async def test_indeterminate_timeout_maps_to_timeout_verdict():
    """``IndeterminateResult`` with a timeout reason -> ``Verdict.timeout``."""
    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(
            IndeterminateResult(reason="Timeout waiting for cvc5 after 30s")
        ),
    )
    assert rr.underlying_verdict == Verdict.timeout
    assert rr.smt_script is not None


@pytest.mark.asyncio
async def test_indeterminate_other_maps_to_unknown_verdict():
    """``IndeterminateResult`` with a non-timeout reason -> ``Verdict.unknown``."""
    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(
            IndeterminateResult(reason="cvc5 returned unknown")
        ),
    )
    assert rr.underlying_verdict == Verdict.unknown
    assert rr.smt_script is not None
    assert "cvc5 returned unknown" in rr.reason


# --- Planner failure paths --------------------------------------------


@pytest.mark.asyncio
async def test_planner_failure_short_circuits_to_planner_failed():
    """``TextToSQLFailure`` becomes ``planner_failed`` and skips equivalence."""
    eq_called = False

    async def _check(*args, **kwargs):
        nonlocal eq_called
        eq_called = True
        return EquivalentResult()

    failure = TextToSQLFailure(error="bedrock unavailable", code=ErrorCode.LLM_API_ERROR)
    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(failure),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_check,
    )
    assert rr.underlying_verdict == Verdict.planner_failed
    assert rr.smt_script is None
    assert "LLM_API_ERROR" in rr.reason
    assert "bedrock unavailable" in rr.reason
    assert eq_called is False


@pytest.mark.asyncio
async def test_planner_timeout_uses_planner_timeout_reason():
    """A planner that exceeds the per-Test_Case budget -> reason 'planner_timeout'."""

    async def _slow_planner(**kwargs):
        await asyncio.sleep(5)
        return _make_planner_success()

    options = RunOptions(
        bird_root=Path("/tmp/bird"),
        split="dev",
        cvc5_timeout_seconds=30,
        per_test_timeout_seconds=1,  # tight budget so the test is fast
    )
    rr = await run_one(
        _make_test_case(),
        options,
        expected_fail=set(),
        planner_callable=_slow_planner,
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
    )
    assert rr.underlying_verdict == Verdict.planner_failed
    assert rr.reason == "planner_timeout"
    assert rr.error_code == "planner_timeout"


@pytest.mark.asyncio
async def test_planner_empty_message_substitutes_no_message():
    """An empty planner message becomes ``no message`` per Req 3.3."""
    failure = TextToSQLFailure(error="", code=ErrorCode.LLM_API_ERROR)
    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(failure),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
    )
    assert "no message" in rr.reason


# --- Converter failure paths ------------------------------------------


@pytest.mark.asyncio
async def test_converter_unsupported_feature_maps_to_gold_conversion_failure():
    """Gold ``unsupported_feature`` -> ``gold_conversion_failure`` (Req 5.8)."""
    err = ConverterError(
        kind="unsupported_feature",
        message="LEFT JOIN not supported",
        feature="outer_join",
        line=2,
        column=5,
    )
    eq_called = False

    async def _check(*args, **kwargs):
        nonlocal eq_called
        eq_called = True
        return EquivalentResult()

    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(err),
        check_equivalence_callable=_check,
    )
    assert rr.underlying_verdict == Verdict.gold_conversion_failure
    assert "outer_join" in rr.reason
    assert eq_called is False


@pytest.mark.asyncio
async def test_converter_parse_error_maps_to_skipped():
    """Gold ``parse_error`` -> ``Verdict.skipped`` (Req 3.2)."""
    err = ConverterError(
        kind="parse_error",
        message="unexpected token",
        line=1,
        column=10,
    )
    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(err),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
    )
    assert rr.underlying_verdict == Verdict.skipped
    assert "unexpected token" in rr.reason


# --- Reason length cap and substitution -------------------------------


@pytest.mark.asyncio
async def test_reason_truncated_to_2000_chars():
    """Reason field is capped at 2000 characters at the call site (Req 3.5)."""
    long_message = "x" * 5000
    failure = TextToSQLFailure(error=long_message, code=ErrorCode.LLM_API_ERROR)
    rr = await run_one(
        _make_test_case(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(failure),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
    )
    assert len(rr.reason) <= 2000


# --- Expected_Fail override -------------------------------------------


@pytest.mark.asyncio
async def test_expected_fail_overrides_non_equivalent_to_expected_fail():
    """A listed Test_Case with non-equivalent underlying -> reported expected_fail."""
    rr = await run_one(
        _make_test_case(test_case_id="dev_42"),
        _make_options(),
        expected_fail={"dev_42"},
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(NotEquivalentResult()),
    )
    assert rr.underlying_verdict == Verdict.not_equivalent
    assert rr.reported_verdict == Verdict.expected_fail


@pytest.mark.asyncio
async def test_expected_fail_does_not_override_equivalent():
    """A listed Test_Case with equivalent underlying keeps reported equivalent."""
    rr = await run_one(
        _make_test_case(test_case_id="dev_42"),
        _make_options(),
        expected_fail={"dev_42"},
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
    )
    assert rr.underlying_verdict == Verdict.equivalent
    assert rr.reported_verdict == Verdict.equivalent


@pytest.mark.asyncio
async def test_expected_fail_unlisted_test_case_unaffected():
    """Test_Cases not on the list pass through unchanged."""
    rr = await run_one(
        _make_test_case(test_case_id="dev_99"),
        _make_options(),
        expected_fail={"dev_42"},
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(NotEquivalentResult()),
    )
    assert rr.underlying_verdict == Verdict.not_equivalent
    assert rr.reported_verdict == Verdict.not_equivalent


# --- Planner_input recording ------------------------------------------


@pytest.mark.asyncio
async def test_planner_input_records_question_when_no_evidence():
    """Empty evidence -> planner_input == question (Req 11.4)."""
    rr = await run_one(
        _make_test_case(evidence=""),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
    )
    assert rr.planner_input == "How many people are there?"


@pytest.mark.asyncio
async def test_planner_input_records_concatenated_string_with_evidence():
    """With evidence and no dedicated kwarg, planner_input is question\\nevidence."""
    rr = await run_one(
        _make_test_case(evidence="people are humans"),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
    )
    assert rr.planner_input == "How many people are there?\npeople are humans"
