"""Robustness tests for the question converter's LLM-output handling.

Two failure modes the converter has to handle when the model misbehaves:

1. The model emits multiple candidate S-expressions in one response,
   separated by chain-of-thought prose ("Wait, let me redo this...").
   The first candidate parses, the trailing prose then chokes the
   parser. Fix A: extract the *last* balanced top-level S-expression.

2. The parser succeeds but the free-variable validator catches a
   semantic bug (a variable referenced inside a membership tuple but
   not bound by any enclosing ``exists``). The retry hint should
   name the offending variables, not produce a generic "re-balance
   parens" message that doesn't apply.

These tests pin both fixes directly on the helper functions; the
end-to-end retry-prompt wording is exercised by an additional case
that drives ``convert_question`` with a stub LLM that emits the
multi-candidate output we saw in dev_1520.
"""

from __future__ import annotations

import pytest

from text_to_sql_planner.converter.question_converter import (
    _extract_last_sexpr,
    _free_vars_from_validator_error,
    _retry_hint_for,
    _validate_free_variables,
)
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    LogicalConnectiveNode,
    MembershipNode,
    QuantifierNode,
    ComparisonNode,
    LiteralNode,
    VariableRefNode,
)


# ---------------------------------------------------------------------------
# Fix A: _extract_last_sexpr
# ---------------------------------------------------------------------------


def test_extract_last_sexpr_returns_input_for_single_form():
    """A single S-expression is returned verbatim — this helper is a no-op."""
    text = "(drc (a b) (in (a b) T))"
    assert _extract_last_sexpr(text) == text


def test_extract_last_sexpr_returns_input_when_no_form_present():
    """No balanced top-level form → return input as-is so the parser surfaces its usual error."""
    text = "this is just prose with no parens at all"
    assert _extract_last_sexpr(text) == text


def test_extract_last_sexpr_picks_last_of_multiple_candidates():
    """The dev_1520 case: two candidates separated by prose."""
    text = """\
(drc (Date) (in (Date) T))

Wait, let me redo this more carefully.

(drc (Date Consumption) (in (Date Consumption) yearmonth))
"""
    extracted = _extract_last_sexpr(text)
    assert extracted == "(drc (Date Consumption) (in (Date Consumption) yearmonth))"


def test_extract_last_sexpr_ignores_parens_inside_strings():
    """Parens inside string literals must not confuse the depth tracker."""
    text = '(drc (x) (= x "foo)bar)("))'
    # This is one balanced form despite the parens inside the string.
    assert _extract_last_sexpr(text) == text


def test_extract_last_sexpr_handles_escaped_quotes():
    """Backslash-escaped quotes inside a string don't end the string."""
    text = r'(drc (x) (= x "say \"hi\""))'
    assert _extract_last_sexpr(text) == text


def test_extract_last_sexpr_picks_last_with_three_candidates():
    """Third time's the charm — and the helper picks the last one."""
    text = """\
(first attempt with bug)
prose
(second attempt with different bug)
prose
(third attempt that actually works)
"""
    assert _extract_last_sexpr(text) == "(third attempt that actually works)"


# ---------------------------------------------------------------------------
# Fix B: _retry_hint_for and _free_vars_from_validator_error
# ---------------------------------------------------------------------------


def test_free_vars_extractor_pulls_names_from_validator_error():
    """The validator's bracketed list of free vars is parsed via ast.literal_eval."""
    err = (
        "Unquantified free variables: ['Date2', 'Price']. "
        "These must be wrapped in (exists ...) ..."
    )
    assert _free_vars_from_validator_error(err) == ["Date2", "Price"]


def test_free_vars_extractor_returns_empty_for_unrelated_errors():
    """Parser errors and other messages don't trigger the extractor."""
    assert _free_vars_from_validator_error(
        "ParseError at offset 463: Unexpected trailing token: SYMBOL"
    ) == []
    assert _free_vars_from_validator_error("") == []
    assert _free_vars_from_validator_error("Unquantified free variables") == []


def test_free_vars_extractor_handles_nested_brackets_in_message():
    """The depth-tracker walks balanced brackets, not the first ``]`` it sees."""
    # Pathological: a message that contains a ``]`` *after* the actual list.
    err = (
        "Unquantified free variables: ['x', 'y']. "
        "Result variables are: ['z']."
    )
    assert _free_vars_from_validator_error(err) == ["x", "y"]


def test_retry_hint_names_free_variables_when_validator_failed():
    """The dev_1520 hint should literally name the missing binders."""
    err = (
        "Unquantified free variables: ['Date2', 'Price']. "
        "These must be wrapped in (exists ...) ..."
    )
    hint = _retry_hint_for("", err)
    assert "Date2" in hint
    assert "Price" in hint
    # And it should explicitly mention adding them to an existential.
    assert "exists" in hint


def test_retry_hint_falls_back_to_unary_and_or_check():
    """When no validator error is present, the unary-and/or hint still fires."""
    raw = "(drc (x) (and (in (x) T)))"  # unary AND with one operand
    hint = _retry_hint_for(raw, "ParseError at offset 5")
    assert "unary" in hint or "single operand" in hint


def test_retry_hint_falls_back_to_paren_balance_check():
    """A truly mismatched-parens output gets the balance-mismatch hint."""
    raw = "(drc (x) (in (x) T)"  # missing close
    hint = _retry_hint_for(raw, "ParseError")
    assert "paren" in hint.lower()


# ---------------------------------------------------------------------------
# Fix B: validator integration — confirm the dev_1520 candidate fails
# ---------------------------------------------------------------------------


def test_validator_catches_dev_1520_free_variables():
    """The third attempt in dev_1520 — Date2/Price unbound — triggers the validator."""
    # (drc (Date Consumption)
    #   (exists (CustomerID)
    #     (and (exists (TransactionID Time CardID GasStationID ProductID Amount)
    #            (and (in (TransactionID Date2 Time CustomerID CardID GasStationID
    #                      ProductID Amount Price) transactions_1k)
    #                 (= Date2 "2012-08-24") (= Price 124.05)))
    #          (exists (seg cur)
    #            (and (in (CustomerID seg cur) customers)
    #                 (in (CustomerID Date Consumption) yearmonth)
    #                 (= Date "201201"))))))
    inner_exists = QuantifierNode(
        kind="exists",
        variables=["TransactionID", "Time", "CardID", "GasStationID",
                   "ProductID", "Amount"],
        body=LogicalConnectiveNode(
            operator="and",
            left=LogicalConnectiveNode(
                operator="and",
                left=MembershipNode(
                    variables=["TransactionID", "Date2", "Time", "CustomerID",
                               "CardID", "GasStationID", "ProductID",
                               "Amount", "Price"],
                    relation="transactions_1k",
                ),
                right=ComparisonNode(
                    operator="=",
                    left=VariableRefNode(name="Date2"),
                    right=LiteralNode(data_type="string", value="2012-08-24"),
                ),
            ),
            right=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="Price"),
                right=LiteralNode(data_type="number", value=124.05),
            ),
        ),
    )
    customers_exists = QuantifierNode(
        kind="exists",
        variables=["seg", "cur"],
        body=LogicalConnectiveNode(
            operator="and",
            left=MembershipNode(
                variables=["CustomerID", "seg", "cur"], relation="customers"
            ),
            right=LogicalConnectiveNode(
                operator="and",
                left=MembershipNode(
                    variables=["CustomerID", "Date", "Consumption"],
                    relation="yearmonth",
                ),
                right=ComparisonNode(
                    operator="=",
                    left=VariableRefNode(name="Date"),
                    right=LiteralNode(data_type="string", value="201201"),
                ),
            ),
        ),
    )
    expr = DRCExpression(
        result_variables=[
            ColumnVariable(name="Date"),
            ColumnVariable(name="Consumption"),
        ],
        condition=QuantifierNode(
            kind="exists",
            variables=["CustomerID"],
            body=LogicalConnectiveNode(
                operator="and",
                left=inner_exists,
                right=customers_exists,
            ),
        ),
    )

    err = _validate_free_variables(expr)
    assert err is not None
    assert "Date2" in err
    assert "Price" in err


# ---------------------------------------------------------------------------
# End-to-end: convert_question with a stub LLM that emits the dev_1520 shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_convert_question_recovers_from_multi_candidate_output(monkeypatch):
    """When the LLM emits two candidates separated by prose, parse the last one."""
    from text_to_sql_planner.converter import question_converter as qc

    # Simulate Claude returning two candidates, only the second of
    # which is well-formed and validates. The first has a deliberately
    # broken arity so even if it parsed successfully it'd fail
    # downstream — that way the test pins extraction of the LAST
    # candidate, not "any candidate that happens to parse".
    multi_candidate = """\
(drc (BAD) (in (BAD WRONG) does_not_exist))

Wait, let me redo this more carefully.

(drc (id) (exists (name) (in (id name) employees)))
"""

    async def fake_convert(question, schema, config):
        return multi_candidate

    async def fake_distinct(question, schema, config):
        from text_to_sql_planner.planner.llm_client import DistinctDecision
        return DistinctDecision(use_distinct=False, reasoning="")

    monkeypatch.setattr(qc, "convert_question_to_drc", fake_convert)
    monkeypatch.setattr(qc, "decide_distinct", fake_distinct)

    result = await qc.convert_question(
        "Find employee ids",
        "CREATE TABLE employees (id INT, name TEXT)",
        max_attempts=1,
    )

    # The conversion should succeed — by picking the second candidate.
    assert isinstance(result, qc.ConversionSuccess)
    # The lisp_syntax should reflect the candidate that was actually parsed.
    assert "BAD" not in result.lisp_syntax
    assert "employees" in result.lisp_syntax
