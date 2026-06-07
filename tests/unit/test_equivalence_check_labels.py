"""Tests for the labelled cvc5 equivalence-check output.

Without labels, every ``### cvc5 equivalence check`` banner in the
run log looks identical, but the function is called from two
distinct contexts:

1. *Planner-internal:* per-iteration check that the relation built
   so far is equivalent to the LLM's target DRC.
2. *BIRD-final:* runner-level check that the planner's generated
   DRC is equivalent to the gold DRC translated from BIRD's
   reference SQL.

A run that confused these is hard to read — a planner-side
``Planning complete!`` followed by a runner-side
``not_equivalent (skipped cvc5)`` looks like contradiction unless
you know which check is which.

These tests pin the labelled output so the boundary between the
two contexts stays visible.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from text_to_sql_planner.equivalence.equivalence_checker import (
    _build_equivalence_script,
    check_equivalence,
)
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    MembershipNode,
)


def _drc(name: str = "x", relation: str = "T") -> DRCExpression:
    return DRCExpression(
        result_variables=[ColumnVariable(name=name)],
        condition=MembershipNode(variables=[name], relation=relation),
    )


def test_arity_mismatch_warning_includes_side_labels():
    """The early-exit arity warning surfaces the LHS/RHS hint when supplied."""
    expr1 = DRCExpression(
        result_variables=[ColumnVariable(name="a"), ColumnVariable(name="b")],
        condition=MembershipNode(variables=["a", "b"], relation="T"),
    )
    expr2 = DRCExpression(
        result_variables=[ColumnVariable(name="x")],
        condition=MembershipNode(variables=["x"], relation="T"),
    )

    captured = io.StringIO()

    async def _drive():
        with redirect_stdout(captured):
            return await check_equivalence(
                expr1, expr2,
                lhs_label="generated",
                rhs_label="gold",
            )

    import asyncio
    asyncio.run(_drive())

    output = captured.getvalue()
    assert "Arity mismatch" in output
    assert "LHS = generated" in output
    assert "RHS = gold" in output


def test_arity_mismatch_warning_omits_label_block_without_labels():
    """When no labels are supplied the existing format is unchanged."""
    expr1 = DRCExpression(
        result_variables=[ColumnVariable(name="a"), ColumnVariable(name="b")],
        condition=MembershipNode(variables=["a", "b"], relation="T"),
    )
    expr2 = DRCExpression(
        result_variables=[ColumnVariable(name="x")],
        condition=MembershipNode(variables=["x"], relation="T"),
    )

    captured = io.StringIO()

    async def _drive():
        with redirect_stdout(captured):
            return await check_equivalence(expr1, expr2)

    import asyncio
    asyncio.run(_drive())

    output = captured.getvalue()
    assert "Arity mismatch" in output
    # No LHS/RHS hint — pre-fix format preserved for callers that
    # don't pass labels.
    assert "LHS =" not in output
    assert "RHS =" not in output


def test_build_equivalence_script_includes_lhs_rhs_comment_when_labelled():
    """The generated SMT-LIB script gets a comment line naming each side."""
    script = _build_equivalence_script(
        _drc(name="a"), _drc(name="b"),
        lhs_label="generated", rhs_label="gold",
    )
    assert ";; (= LHS RHS)  --  LHS = generated, RHS = gold" in script


def test_build_equivalence_script_omits_comment_without_labels():
    """No labels → no ``;; (= LHS RHS)`` comment."""
    script = _build_equivalence_script(_drc(name="a"), _drc(name="b"))
    assert "LHS = " not in script
    assert "RHS = " not in script


def test_build_equivalence_script_still_well_formed_with_labels():
    """Adding the comment line doesn't break the script structure."""
    script = _build_equivalence_script(
        _drc(name="a"), _drc(name="b"),
        lhs_label="lhs", rhs_label="rhs",
    )
    # Comment comes before the assert; check-sat is at the end.
    comment_idx = script.find(";; (= LHS RHS)")
    assert_idx = script.find("(assert (not")
    check_sat_idx = script.find("(check-sat)")
    assert 0 <= comment_idx < assert_idx < check_sat_idx
