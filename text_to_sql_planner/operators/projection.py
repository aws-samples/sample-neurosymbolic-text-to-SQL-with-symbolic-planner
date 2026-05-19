"""Projection (π) operator: selects specific columns from a relation.

Projects a relation onto a subset of its columns. Columns not in the
projection are existentially quantified in the output condition.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    QuantifierNode,
    MembershipNode,
    ResultVariable,
)
from text_to_sql_planner.types.operators import (
    OperatorResult,
    OperatorSuccess,
    OperatorFailure,
    ProjectionParams,
)


def _get_columns(expr: DRCExpression) -> list[str]:
    """Extract column names from a DRC expression's result variables."""
    return [rv.name if isinstance(rv, ColumnVariable) else rv.column for rv in expr.result_variables]


def apply_projection(params: ProjectionParams, inputs: list[DRCExpression]) -> OperatorResult:
    """Apply projection operator π_{columns}(R).

    Takes a list of columns and one input relation. Validates all columns
    exist in the input. Output contains only the specified columns, with
    removed columns wrapped in an exists quantifier.
    """
    if len(inputs) != 1:
        return OperatorFailure(error="Projection requires exactly 1 input relation")

    relation = inputs[0]
    columns = params.columns

    if not columns:
        return OperatorFailure(error="Projection requires at least one column")

    input_columns = _get_columns(relation)

    # Validate all projection columns exist in input
    for col in columns:
        if col not in input_columns:
            return OperatorFailure(
                error=f"Projection column '{col}' not found in input relation. "
                f"Available columns: {input_columns}"
            )

    # Output variables: only the projected columns (preserve order from params)
    output_variables: list[ResultVariable] = []
    for rv in relation.result_variables:
        col_name = rv.name if isinstance(rv, ColumnVariable) else rv.column
        if col_name in columns:
            output_variables.append(rv)

    # Columns being removed (existentially quantified)
    removed_columns = [col for col in input_columns if col not in columns]

    if not removed_columns:
        # No columns removed, condition stays the same
        output_condition = relation.condition
    else:
        # Wrap with exists for removed columns
        # Find relation name from the input expression
        relation_name = _extract_relation_name(relation)
        output_condition = QuantifierNode(
            kind="exists",
            variables=removed_columns,
            relation=relation_name,
            body=relation.condition,
        )

    output = DRCExpression(
        result_variables=output_variables,
        condition=output_condition,
    )

    return OperatorSuccess(output=output)


def _extract_relation_name(expr: DRCExpression) -> str:
    """Extract a relation name from a DRC expression's condition."""
    from text_to_sql_planner.types.drc import LogicalConnectiveNode

    condition = expr.condition
    if isinstance(condition, MembershipNode):
        return condition.relation
    if isinstance(condition, QuantifierNode):
        return condition.relation
    return _find_relation_name(condition)


def _find_relation_name(node) -> str:
    """Recursively search for a relation name in a condition tree."""
    from text_to_sql_planner.types.drc import LogicalConnectiveNode

    if node is None:
        return "R"
    if isinstance(node, MembershipNode):
        return node.relation
    if isinstance(node, QuantifierNode):
        return node.relation
    if isinstance(node, LogicalConnectiveNode):
        result = _find_relation_name(node.left)
        if result != "R":
            return result
        return _find_relation_name(node.right)
    return "R"
