"""Relational algebra operator implementations.

Provides the `apply_operator` dispatcher that routes an OperatorApplication
to the correct operator implementation based on the operator type.
"""

from __future__ import annotations

from text_to_sql_planner.types.operators import (
    OperatorApplication,
    OperatorResult,
    OperatorSuccess,
    OperatorFailure,
    SelectionParams,
    JoinParams,
    ProjectionParams,
    CartesianProductParams,
    UnionParams,
    DifferenceParams,
    DivisionParams,
)
from text_to_sql_planner.operators.selection import apply_selection
from text_to_sql_planner.operators.join import apply_join
from text_to_sql_planner.operators.projection import apply_projection
from text_to_sql_planner.operators.cartesian_product import apply_cartesian_product
from text_to_sql_planner.operators.union import apply_union
from text_to_sql_planner.operators.difference import apply_difference
from text_to_sql_planner.operators.division import apply_division


__all__ = [
    "apply_operator",
    "apply_selection",
    "apply_join",
    "apply_projection",
    "apply_cartesian_product",
    "apply_union",
    "apply_difference",
    "apply_division",
]


# Operators that require exactly 1 input relation
_UNARY_OPERATORS = {"selection", "projection"}

# Operators that require exactly 2 input relations
_BINARY_OPERATORS = {"join", "cartesian_product", "union", "difference", "division"}

_KNOWN_OPERATORS = _UNARY_OPERATORS | _BINARY_OPERATORS


def apply_operator(application: OperatorApplication) -> OperatorResult:
    """Dispatch an operator application to the correct implementation.

    Routes based on `application.operator` field and validates that the
    correct number of inputs are provided for each operator type.

    Args:
        application: The operator application containing operator type,
                     parameters, and input relations.

    Returns:
        OperatorSuccess with the output DRCExpression, or
        OperatorFailure with a descriptive error message.
    """
    operator = application.operator
    inputs = application.inputs
    params = application.params

    # Validate operator type
    if operator not in _KNOWN_OPERATORS:
        return OperatorFailure(error=f"Unknown operator type: '{operator}'")

    # Validate input count
    if operator in _UNARY_OPERATORS:
        if len(inputs) != 1:
            return OperatorFailure(
                error=f"Operator '{operator}' requires exactly 1 input relation, "
                f"got {len(inputs)}"
            )
    elif operator in _BINARY_OPERATORS:
        if len(inputs) != 2:
            return OperatorFailure(
                error=f"Operator '{operator}' requires exactly 2 input relations, "
                f"got {len(inputs)}"
            )

    # Dispatch to the correct operator
    if operator == "selection":
        return apply_selection(params, inputs)
    elif operator == "join":
        return apply_join(params, inputs)
    elif operator == "projection":
        return apply_projection(params, inputs)
    elif operator == "cartesian_product":
        return apply_cartesian_product(params, inputs)
    elif operator == "union":
        return apply_union(params, inputs)
    elif operator == "difference":
        return apply_difference(params, inputs)
    elif operator == "division":
        return apply_division(params, inputs)

    # Should never reach here due to validation above
    return OperatorFailure(error=f"Unhandled operator: '{operator}'")
