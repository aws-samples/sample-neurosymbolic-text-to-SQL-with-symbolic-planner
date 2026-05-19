"""Selection (σ) operator: filters tuples based on a condition.

Applies a selection condition to an input relation. The output has the same
columns as the input, with the condition conjoined (AND) with the selection
condition.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    DRCExpression,
    LogicalConnectiveNode,
)
from text_to_sql_planner.types.operators import (
    OperatorResult,
    OperatorSuccess,
    OperatorFailure,
    SelectionParams,
)


def apply_selection(params: SelectionParams, inputs: list[DRCExpression]) -> OperatorResult:
    """Apply selection operator σ_condition(R).

    Takes a condition and one input relation. Output has the same columns
    as the input, with the condition being (and original_condition selection_condition).
    """
    if len(inputs) != 1:
        return OperatorFailure(error="Selection requires exactly 1 input relation")

    relation = inputs[0]

    if params.condition is None:
        return OperatorFailure(error="Selection requires a condition")

    # Output columns are the same as input
    output_variables = list(relation.result_variables)

    # Output condition: (and original_condition selection_condition)
    output_condition = LogicalConnectiveNode(
        operator="and",
        left=relation.condition,
        right=params.condition,
    )

    output = DRCExpression(
        result_variables=output_variables,
        condition=output_condition,
    )

    return OperatorSuccess(output=output)
