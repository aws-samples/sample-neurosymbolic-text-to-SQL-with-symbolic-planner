"""Set difference (−) operator: tuples in R that are not in S.

Produces ``R − S`` over two union-compatible relations (same arity).
The output keeps R's columns and result variables; the output condition
is::

    C_R(x1, ..., xn) ∧ ¬ C_S(x1, ..., xn)

Where S references columns under different names, those references are
alpha-renamed positionally to R's column names so the inner negated
condition speaks the same variable language as the outer projection.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    LogicalConnectiveNode,
    NotNode,
)
from text_to_sql_planner.operators._rename import rename_free
from text_to_sql_planner.operators.exactly_n import (
    emit_canonical_exactly_n,
    recognize_ge_n_pattern,
)
from text_to_sql_planner.types.operators import (
    DifferenceParams,
    OperatorFailure,
    OperatorResult,
    OperatorSuccess,
)


def _get_columns(expr: DRCExpression) -> list[str]:
    """Extract column names from a DRC expression's result variables."""
    return [
        rv.name if isinstance(rv, ColumnVariable) else rv.column
        for rv in expr.result_variables
    ]


def apply_difference(
    params: DifferenceParams, inputs: list[DRCExpression]
) -> OperatorResult:
    """Apply set difference operator R − S.

    Takes two input relations. Validates union-compatibility (same arity).
    Output columns are R's. Output condition is

        C_R(x1..xn) ∧ ¬ C_S(x1..xn)

    where S's free variables are positionally alpha-renamed to R's
    column names so the negated subformula speaks about R's tuples.
    """
    if len(inputs) != 2:
        return OperatorFailure(error="Difference requires exactly 2 input relations")

    r1, r2 = inputs[0], inputs[1]

    if len(r1.result_variables) != len(r2.result_variables):
        return OperatorFailure(
            error=(
                f"Difference requires relations with the same arity. "
                f"First relation has {len(r1.result_variables)} columns, "
                f"second has {len(r2.result_variables)} columns."
            )
        )

    r1_cols = _get_columns(r1)
    r2_cols = _get_columns(r2)

    # Special-case ``≥N − ≥(N+1)`` for the same relation: emit the
    # canonical "exactly N" form. This both produces a clean DRC for
    # the LLM and SQL converter and aligns Skolem terms across both
    # sides of an equivalence check (see
    # :mod:`text_to_sql_planner.operators.exactly_n` for soundness
    # discussion).
    p1 = recognize_ge_n_pattern(r1)
    p2 = recognize_ge_n_pattern(r2)
    if (
        p1 is not None
        and p2 is not None
        and p1.relation == p2.relation
        and p1.witness_slot == p2.witness_slot
        and p1.key_slots == p2.key_slots
        and p1.slot_arity == p2.slot_arity
        and len(p2.witnesses) == len(p1.witnesses) + 1
    ):
        return OperatorSuccess(output=emit_canonical_exactly_n(r1, p1))

    # Positional alpha-rename: every free reference to r2's i-th column
    # name becomes r1's i-th column name. If the names already match
    # this is a no-op for that position.
    rename_map: dict[str, str] = {
        old: new for old, new in zip(r2_cols, r1_cols) if old != new
    }
    r2_renamed_condition = (
        rename_free(r2.condition, rename_map) if rename_map else r2.condition
    )

    output = DRCExpression(
        result_variables=list(r1.result_variables),
        condition=LogicalConnectiveNode(
            operator="and",
            left=r1.condition,
            right=NotNode(operand=r2_renamed_condition),
        ),
    )

    return OperatorSuccess(output=output)
