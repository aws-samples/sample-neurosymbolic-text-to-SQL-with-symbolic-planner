"""Projection (π) operator: selects specific columns from a relation.

Projects a relation onto a subset of its columns. Columns not in the
projection are existentially quantified in the output condition.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    AggregateVariable,
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

# Aggregate function names
_AGGREGATE_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


def _get_columns(expr: DRCExpression) -> list[str]:
    """Extract column names from a DRC expression's result variables."""
    return [rv.name if isinstance(rv, ColumnVariable) else rv.column for rv in expr.result_variables]


def _parse_column_spec(col: str) -> ResultVariable:
    """Parse a column specification which may be a plain name or an aggregate.

    Handles formats like:
    - "name" → ColumnVariable(name="name")
    - "COUNT s_id" → AggregateVariable(function="COUNT", column="s_id")
    - "COUNT(s_id)" → AggregateVariable(function="COUNT", column="s_id")
    - "(COUNT s_id)" → AggregateVariable(function="COUNT", column="s_id")
    - "AVG grade" → AggregateVariable(function="AVG", column="grade")
    """
    col = col.strip()

    # Check for "(FUNC col)" format (Lisp-style with outer parens)
    if col.startswith("(") and col.endswith(")"):
        inner = col[1:-1].strip()
        parts = inner.split()
        if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
            return AggregateVariable(function=parts[0], column=parts[1])

    # Check for "FUNC(col)" format
    for func in _AGGREGATE_FUNCS:
        if col.startswith(f"{func}(") and col.endswith(")"):
            inner = col[len(func) + 1:-1].strip()
            return AggregateVariable(function=func, column=inner)

    # Check for "FUNC col" format (space-separated)
    parts = col.split()
    if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
        return AggregateVariable(function=parts[0], column=parts[1])

    return ColumnVariable(name=col)


def _get_underlying_column(col_spec: str) -> str:
    """Get the underlying column name from a column spec (handles aggregates)."""
    rv = _parse_column_spec(col_spec)
    if isinstance(rv, AggregateVariable):
        return rv.column
    return rv.name


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

    # Validate all projection columns — for aggregates, check the underlying column
    for col in columns:
        underlying = _get_underlying_column(col)
        if underlying not in input_columns:
            return OperatorFailure(
                error=f"Projection column '{col}' (underlying: '{underlying}') not found in input relation. "
                f"Available columns: {input_columns}"
            )

    # Build output variables from the column specs
    output_variables: list[ResultVariable] = []
    projected_underlying: list[str] = []
    for col in columns:
        rv = _parse_column_spec(col)
        output_variables.append(rv)
        if isinstance(rv, AggregateVariable):
            projected_underlying.append(rv.column)
        else:
            projected_underlying.append(rv.name)

    # Columns being removed (existentially quantified)
    removed_columns = [c for c in input_columns if c not in projected_underlying]

    # If all output variables are aggregates, the condition stays as-is
    # (aggregates operate over all rows matching the condition)
    all_aggregates = all(isinstance(rv, AggregateVariable) for rv in output_variables)

    if not removed_columns or all_aggregates:
        # No columns removed, or all outputs are aggregates — condition stays the same
        output_condition = relation.condition
    else:
        # Wrap with exists quantifier for the REMOVED columns only.
        # Result variables are free and must NOT be quantified.
        # The body includes the original condition (membership + any filters).
        output_condition = QuantifierNode(
            kind="exists",
            variables=removed_columns,
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
    return _find_relation_name(condition)


def _find_relation_name(node) -> str:
    """Recursively search for a relation name in a condition tree."""
    from text_to_sql_planner.types.drc import LogicalConnectiveNode

    if node is None:
        return "R"
    if isinstance(node, MembershipNode):
        return node.relation
    if isinstance(node, QuantifierNode):
        return _find_relation_name(node.body)
    if isinstance(node, LogicalConnectiveNode):
        result = _find_relation_name(node.left)
        if result != "R":
            return result
        return _find_relation_name(node.right)
    return "R"
