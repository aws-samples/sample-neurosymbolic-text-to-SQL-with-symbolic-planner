"""Cartesian Product (×) operator: combines all tuples from two relations.

Produces the cross product of two relations. Output columns are the
concatenation of both input column lists, and the condition is the
conjunction (AND) of both input conditions.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    DRCExpression,
    LogicalConnectiveNode,
    ResultVariable,
)
from text_to_sql_planner.types.operators import (
    OperatorResult,
    OperatorSuccess,
    OperatorFailure,
    CartesianProductParams,
)


def apply_cartesian_product(
    params: CartesianProductParams, inputs: list[DRCExpression]
) -> OperatorResult:
    """Apply cartesian product operator R × S.

    Takes two input relations. Output columns are the concatenation of both
    input column lists. Output condition is (and r1_condition r2_condition).
    """
    if len(inputs) != 2:
        return OperatorFailure(error="Cartesian product requires exactly 2 input relations")

    r1, r2 = inputs[0], inputs[1]

    # Output columns: concatenation of both inputs
    output_variables: list[ResultVariable] = list(r1.result_variables) + list(r2.result_variables)

    # Output condition: (and r1_condition r2_condition)
    output_condition = LogicalConnectiveNode(
        operator="and",
        left=r1.condition,
        right=r2.condition,
    )

    output = DRCExpression(
        result_variables=output_variables,
        condition=output_condition,
    )

    return OperatorSuccess(output=output)
