"""Join (⋈) operator: natural join on specified columns.

Joins two relations on shared column names. The output contains the union
of columns from both inputs, with join columns appearing only once.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    LogicalConnectiveNode,
    ComparisonNode,
    QuantifierNode,
    VariableRefNode,
    ResultVariable,
)
from text_to_sql_planner.types.operators import (
    OperatorResult,
    OperatorSuccess,
    OperatorFailure,
    JoinParams,
)


def _get_columns(expr: DRCExpression) -> list[str]:
    """Extract column names from a DRC expression's result variables."""
    return [rv.name if isinstance(rv, ColumnVariable) else rv.column for rv in expr.result_variables]


def apply_join(params: JoinParams, inputs: list[DRCExpression]) -> OperatorResult:
    """Apply join operator R ⋈_{join_columns} S.

    Takes join_columns and two input relations. Validates that join columns
    exist in both inputs. Output columns are the union of both inputs with
    join columns appearing once.
    """
    if len(inputs) != 2:
        return OperatorFailure(error="Join requires exactly 2 input relations")

    r1, r2 = inputs[0], inputs[1]
    join_columns = params.join_columns

    if not join_columns:
        return OperatorFailure(error="Join requires at least one join column")

    r1_columns = _get_columns(r1)
    r2_columns = _get_columns(r2)

    # Validate join columns exist in both inputs
    for col in join_columns:
        if col not in r1_columns:
            return OperatorFailure(
                error=f"Join column '{col}' not found in first input relation. "
                f"Available columns: {r1_columns}"
            )
        if col not in r2_columns:
            return OperatorFailure(
                error=f"Join column '{col}' not found in second input relation. "
                f"Available columns: {r2_columns}"
            )

    # Build output columns: all from r1, then non-join columns from r2
    output_variables: list[ResultVariable] = list(r1.result_variables)
    for rv in r2.result_variables:
        col_name = rv.name if isinstance(rv, ColumnVariable) else rv.column
        if col_name not in join_columns:
            output_variables.append(rv)

    # Build equality conditions for join columns
    # Start with the innermost equality
    equality_conditions = []
    for col in join_columns:
        eq = ComparisonNode(
            operator="=",
            left=VariableRefNode(name=col),
            right=VariableRefNode(name=col),
        )
        equality_conditions.append(eq)

    # Combine: (and r2_condition (= col1 col1) (= col2 col2) ...)
    # Build from right to left
    inner = r2.condition
    for eq in equality_conditions:
        inner = LogicalConnectiveNode(operator="and", left=inner, right=eq)

    # Combine with r1_condition: (and r1_condition inner)
    combined = LogicalConnectiveNode(operator="and", left=r1.condition, right=inner)

    # Wrap with exists for r2 variables
    r2_vars = [col for col in r2_columns if col in join_columns]
    # Actually, the exists wraps the variables from r2 that are being quantified
    # The full structure:
    # (exists (vars_from_r2) r2_relation (exists (vars_from_r1) r1_relation (and r1_cond (and r2_cond (= join_col ...)))))
    # But since we're composing symbolically, we build the nested condition directly

    # Build the output condition following the spec:
    # (exists (vars_from_r1) r1_relation (exists (vars_from_r2) r2_relation (and r1_condition (and r2_condition (= join_col_r1 join_col_r2)))))

    # Build innermost: (and r2_condition equality_chain)
    eq_chain = equality_conditions[0]
    for eq in equality_conditions[1:]:
        eq_chain = LogicalConnectiveNode(operator="and", left=eq_chain, right=eq)

    inner_and = LogicalConnectiveNode(operator="and", left=r2.condition, right=eq_chain)

    # (and r1_condition inner_and)
    outer_and = LogicalConnectiveNode(operator="and", left=r1.condition, right=inner_and)

    # Wrap with exists for r2 vars
    r2_exist_vars = r2_columns
    r2_relation_name = _extract_relation_name(r2)

    exists_r2 = QuantifierNode(
        kind="exists",
        variables=r2_exist_vars,
        relation=r2_relation_name,
        body=outer_and,
    )

    # Wrap with exists for r1 vars
    r1_exist_vars = r1_columns
    r1_relation_name = _extract_relation_name(r1)

    output_condition = QuantifierNode(
        kind="exists",
        variables=r1_exist_vars,
        relation=r1_relation_name,
        body=exists_r2,
    )

    output = DRCExpression(
        result_variables=output_variables,
        condition=output_condition,
    )

    return OperatorSuccess(output=output)


def _extract_relation_name(expr: DRCExpression) -> str:
    """Extract a relation name from a DRC expression's condition.

    Looks for a MembershipNode or QuantifierNode to find the relation name.
    Falls back to 'R' if none found.
    """
    from text_to_sql_planner.types.drc import MembershipNode

    condition = expr.condition
    if isinstance(condition, MembershipNode):
        return condition.relation
    if isinstance(condition, QuantifierNode):
        return condition.relation
    # For composed expressions, try to find a membership node
    return _find_relation_name(condition)


def _find_relation_name(node) -> str:
    """Recursively search for a relation name in a condition tree."""
    from text_to_sql_planner.types.drc import MembershipNode

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
