"""Cartesian Product (×) operator: combines all tuples from two relations.

Produces the cross product of two relations. Output columns are the
concatenation of both input column lists. When columns overlap (e.g.,
self-join), suffixes _1 and _2 are added to disambiguate.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCCondition,
    DRCExpression,
    LogicalConnectiveNode,
    MembershipNode,
    VariableRefNode,
    QuantifierNode,
    ComparisonNode,
    ResultVariable,
)
from text_to_sql_planner.types.operators import (
    OperatorResult,
    OperatorSuccess,
    OperatorFailure,
    CartesianProductParams,
)


def _get_columns(expr: DRCExpression) -> list[str]:
    return [rv.name if isinstance(rv, ColumnVariable) else rv.column for rv in expr.result_variables]


def _rename_variable(condition: DRCCondition, old_name: str, new_name: str) -> DRCCondition:
    """Recursively rename a variable in a condition tree."""
    if isinstance(condition, VariableRefNode):
        if condition.name == old_name:
            return VariableRefNode(name=new_name)
        return condition
    elif isinstance(condition, MembershipNode):
        new_vars = [new_name if v == old_name else v for v in condition.variables]
        return MembershipNode(variables=new_vars, relation=condition.relation)
    elif isinstance(condition, LogicalConnectiveNode):
        return LogicalConnectiveNode(
            operator=condition.operator,
            left=_rename_variable(condition.left, old_name, new_name),
            right=_rename_variable(condition.right, old_name, new_name),
        )
    elif isinstance(condition, ComparisonNode):
        return ComparisonNode(
            operator=condition.operator,
            left=_rename_variable(condition.left, old_name, new_name),
            right=_rename_variable(condition.right, old_name, new_name),
        )
    elif isinstance(condition, QuantifierNode):
        new_vars = [new_name if v == old_name else v for v in condition.variables]
        return QuantifierNode(
            kind=condition.kind,
            variables=new_vars,
            body=_rename_variable(condition.body, old_name, new_name),
        )
    return condition


def apply_cartesian_product(
    params: CartesianProductParams, inputs: list[DRCExpression]
) -> OperatorResult:
    """Apply cartesian product operator R × S.

    Takes two input relations. Output columns are the concatenation of both
    input column lists. When columns overlap, suffixes _1 and _2 are added.
    Output condition is (and r1_condition r2_condition).
    """
    if len(inputs) != 2:
        return OperatorFailure(error="Cartesian product requires exactly 2 input relations")

    r1, r2 = inputs[0], inputs[1]
    r1_columns = _get_columns(r1)
    r2_columns = _get_columns(r2)

    # Check for overlapping column names
    overlap = set(r1_columns) & set(r2_columns)

    if not overlap:
        # No conflicts — simple concatenation
        output_variables: list[ResultVariable] = list(r1.result_variables) + list(r2.result_variables)
        output_condition = LogicalConnectiveNode(
            operator="and",
            left=r1.condition,
            right=r2.condition,
        )
    else:
        # Overlapping columns — rename both sides with _1 and _2 suffixes
        r1_condition = r1.condition
        r2_condition = r2.condition
        r1_renames: dict[str, str] = {}
        r2_renames: dict[str, str] = {}

        for col in overlap:
            r1_renames[col] = f"{col}_1"
            r2_renames[col] = f"{col}_2"

        # Rename in r1
        for old, new in r1_renames.items():
            r1_condition = _rename_variable(r1_condition, old, new)

        # Rename in r2
        for old, new in r2_renames.items():
            r2_condition = _rename_variable(r2_condition, old, new)

        # Build output variables with renamed columns
        output_variables = []
        for rv in r1.result_variables:
            col_name = rv.name if isinstance(rv, ColumnVariable) else rv.column
            if col_name in r1_renames:
                output_variables.append(ColumnVariable(name=r1_renames[col_name]))
            else:
                output_variables.append(rv)

        for rv in r2.result_variables:
            col_name = rv.name if isinstance(rv, ColumnVariable) else rv.column
            if col_name in r2_renames:
                output_variables.append(ColumnVariable(name=r2_renames[col_name]))
            else:
                output_variables.append(rv)

        output_condition = LogicalConnectiveNode(
            operator="and",
            left=r1_condition,
            right=r2_condition,
        )

    output = DRCExpression(
        result_variables=output_variables,
        condition=output_condition,
    )

    return OperatorSuccess(output=output)
