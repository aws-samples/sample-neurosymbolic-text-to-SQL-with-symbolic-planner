"""Ratio operator: produce an ArithmeticResultVariable from two columns.

Takes a relation with exactly 2 ColumnVariable result variables and
produces a single ArithmeticResultVariable wrapping two
AggregateVariables. The condition is preserved unchanged.

When ``params.numerator_condition`` is set, the numerator becomes a
CountIfVariable instead of a plain AggregateVariable.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ArithmeticResultVariable,
    ColumnVariable,
    CountIfVariable,
    DRCExpression,
    QuantifierNode,
    ScalarLiteralVariable,
)
from text_to_sql_planner.types.operators import (
    OperatorFailure,
    OperatorResult,
    OperatorSuccess,
    RatioParams,
)

_AGGREGATE_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


def apply_ratio(params: RatioParams, inputs: list[DRCExpression]) -> OperatorResult:
    """Apply the ratio operator."""

    if len(inputs) != 1:
        return OperatorFailure(error="Ratio requires exactly 1 input relation")

    relation = inputs[0]

    for fn in (params.numerator_function, params.denominator_function):
        if fn not in _AGGREGATE_FUNCS:
            return OperatorFailure(error=f"Unsupported aggregate function: '{fn}'")

    if not params.numerator_column or not params.denominator_column:
        return OperatorFailure(error="Ratio requires non-empty numerator_column and denominator_column")

    # All input result variables must be ColumnVariable
    for i, rv in enumerate(relation.result_variables):
        if not isinstance(rv, ColumnVariable):
            return OperatorFailure(
                error=f"Ratio requires all input result variables to be columns. Position {i} is {type(rv).__name__}"
            )

    col_names = [rv.name for rv in relation.result_variables]
    if params.numerator_column not in col_names:
        return OperatorFailure(
            error=f"Numerator column '{params.numerator_column}' not in input columns {col_names}"
        )
    if params.denominator_column not in col_names:
        return OperatorFailure(
            error=f"Denominator column '{params.denominator_column}' not in input columns {col_names}"
        )

    # Build numerator: either CountIfVariable (with condition) or AggregateVariable
    if params.numerator_condition:
        from text_to_sql_planner.parser.parser import parse_condition
        cond_result = parse_condition(params.numerator_condition)
        if cond_result is None:
            return OperatorFailure(
                error=f"Failed to parse numerator_condition: '{params.numerator_condition}'"
            )
        numerator = CountIfVariable(condition=cond_result, column=params.numerator_column)
    else:
        numerator = AggregateVariable(function=params.numerator_function, column=params.numerator_column)

    # Build denominator: either CountIfVariable (with condition) or AggregateVariable
    if params.denominator_condition:
        from text_to_sql_planner.parser.parser import parse_condition
        cond_result = parse_condition(params.denominator_condition)
        if cond_result is None:
            return OperatorFailure(
                error=f"Failed to parse denominator_condition: '{params.denominator_condition}'"
            )
        denominator = CountIfVariable(condition=cond_result, column=params.denominator_column)
    else:
        denominator = AggregateVariable(function=params.denominator_function, column=params.denominator_column)

    # The ratio operator produces a scalar aggregate result. All the
    # input's column variables are now aggregated over — they must be
    # existentially quantified in the output condition so they don't
    # appear as free constants in the SMT formula.
    bound_vars = [rv.name for rv in relation.result_variables if isinstance(rv, ColumnVariable)]
    condition = relation.condition
    if bound_vars:
        condition = QuantifierNode(kind="exists", variables=bound_vars, body=condition)

    output = DRCExpression(
        result_variables=[
            ArithmeticResultVariable(
                operator=params.operator,
                left=numerator,
                right=denominator,
            )
        ],
        condition=condition,
    )
    # Wrap in scalar multiplication if requested (e.g. * 100 for percentage)
    if params.scalar_multiplier is not None:
        inner = output.result_variables[0]
        output = DRCExpression(
            result_variables=[
                ArithmeticResultVariable(
                    operator="*",
                    left=inner,
                    right=ScalarLiteralVariable(value=params.scalar_multiplier),
                )
            ],
            condition=condition,
        )
    return OperatorSuccess(output=output)
