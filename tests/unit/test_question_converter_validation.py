"""Tests for the scope-aware free-variable validator in
``text_to_sql_planner/converter/question_converter.py``.

The validator is invoked after the LLM emits a DRC expression, before
the planner accepts it. It catches the dangling-binder pattern where a
variable is bound by an ``(exists ...)`` whose scope is too narrow,
leaving downstream references to that name as free constants.
"""

from __future__ import annotations

from text_to_sql_planner.converter.question_converter import (
    _validate_free_variables,
)
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    ComparisonNode,
    DRCExpression,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)


# ---------------------------------------------------------------------------
# Cases the validator should accept
# ---------------------------------------------------------------------------


def test_well_scoped_passes():
    """``∃ r1. PR(r1, x) ∧ ∃ r2. PR(r2, x) ∧ r1 != r2`` is valid."""
    inner = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["r2", "x"], relation="PR"),
        right=ComparisonNode(
            operator="!=",
            left=VariableRefNode(name="r1"),
            right=VariableRefNode(name="r2"),
        ),
    )
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["r1", "x"], relation="PR"),
        right=QuantifierNode(kind="exists", variables=["r2"], body=inner),
    )
    expr = DRCExpression(
        result_variables=[ColumnVariable(name="x")],
        condition=QuantifierNode(kind="exists", variables=["r1"], body=body),
    )

    assert _validate_free_variables(expr) is None


def test_result_variable_is_free_ok():
    """Result variables are allowed to be free."""
    expr = DRCExpression(
        result_variables=[ColumnVariable(name="x")],
        condition=MembershipNode(variables=["x"], relation="R"),
    )
    assert _validate_free_variables(expr) is None


# ---------------------------------------------------------------------------
# Cases the validator should reject — scope-aware analysis
# ---------------------------------------------------------------------------


def test_dangling_binder_reference_after_scope_close():
    """The classic "exactly N" bug: ``r1`` and ``r2`` bound by
    existentials whose scope ends before the negated clause that
    references them.

    Shape:
        (and (exists (r1) (exists (r2) (PR(r1) ∧ PR(r2) ∧ r1≠r2)))
             (not (exists (r3) (PR(r3) ∧ r3≠r1 ∧ r3≠r2))))

    The ``r1`` and ``r2`` inside ``(not ...)`` are free constants.
    """
    positive = QuantifierNode(
        kind="exists",
        variables=["r1"],
        body=QuantifierNode(
            kind="exists",
            variables=["r2"],
            body=LogicalConnectiveNode(
                operator="and",
                left=LogicalConnectiveNode(
                    operator="and",
                    left=MembershipNode(variables=["r1", "x"], relation="PR"),
                    right=MembershipNode(variables=["r2", "x"], relation="PR"),
                ),
                right=ComparisonNode(
                    operator="!=",
                    left=VariableRefNode(name="r1"),
                    right=VariableRefNode(name="r2"),
                ),
            ),
        ),
    )
    negative = NotNode(
        operand=QuantifierNode(
            kind="exists",
            variables=["r3"],
            body=LogicalConnectiveNode(
                operator="and",
                left=MembershipNode(variables=["r3", "x"], relation="PR"),
                right=LogicalConnectiveNode(
                    operator="and",
                    left=ComparisonNode(
                        operator="!=",
                        left=VariableRefNode(name="r3"),
                        right=VariableRefNode(name="r1"),
                    ),
                    right=ComparisonNode(
                        operator="!=",
                        left=VariableRefNode(name="r3"),
                        right=VariableRefNode(name="r2"),
                    ),
                ),
            ),
        ),
    )
    expr = DRCExpression(
        result_variables=[ColumnVariable(name="x")],
        condition=LogicalConnectiveNode(
            operator="and", left=positive, right=negative,
        ),
    )

    err = _validate_free_variables(expr)
    assert err is not None
    # The validator should mention r1 and r2 as the dangling references.
    assert "r1" in err
    assert "r2" in err


def test_unquantified_variable_is_free():
    """A variable that's never bound is straightforwardly free."""
    expr = DRCExpression(
        result_variables=[ColumnVariable(name="x")],
        condition=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="x"),
            right=VariableRefNode(name="never_bound"),
        ),
    )
    err = _validate_free_variables(expr)
    assert err is not None
    assert "never_bound" in err
