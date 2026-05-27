"""Union (∪) operator: combines tuples from two union-compatible relations.

Produces the set union of two relations that have the same arity (number
of columns). Output condition is the disjunction (OR) of both input conditions.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    LogicalConnectiveNode,
)
from text_to_sql_planner.types.operators import (
    OperatorResult,
    OperatorSuccess,
    OperatorFailure,
    UnionParams,
)


def _get_columns(expr: DRCExpression) -> list[str]:
    """Extract column names from a DRC expression's result variables."""
    return [rv.name if isinstance(rv, ColumnVariable) else rv.column for rv in expr.result_variables]


def apply_union(params: UnionParams, inputs: list[DRCExpression]) -> OperatorResult:
    """Apply union operator R ∪ S.

    Takes two input relations. Validates they have the same arity (number
    of result variables). Output columns are the same as the inputs.
    Output condition is (or r1_condition r2_condition).
    """
    if len(inputs) != 2:
        return OperatorFailure(error="Union requires exactly 2 input relations")

    r1, r2 = inputs[0], inputs[1]

    # Validate same arity
    if len(r1.result_variables) != len(r2.result_variables):
        return OperatorFailure(
            error=f"Union requires relations with the same arity. "
            f"First relation has {len(r1.result_variables)} columns, "
            f"second has {len(r2.result_variables)} columns."
        )

    # Output columns: same as first input
    output_variables = list(r1.result_variables)

    # Output condition: (or r1_condition r2_condition)
    output_condition = LogicalConnectiveNode(
        operator="or",
        left=r1.condition,
        right=r2.condition,
    )

    output = DRCExpression(
        result_variables=output_variables,
        condition=output_condition,
    )

    return OperatorSuccess(output=output)
