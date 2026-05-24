"""DRC expression simplifier: applies correctness-preserving transformations.

Transformations:
1. Merge nested quantifiers: ∃ X (∃ Y (∃ Z body)) → ∃ X,Y,Z body
2. (More can be added later)
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    DRCExpression,
    DRCCondition,
    QuantifierNode,
    LogicalConnectiveNode,
    NotNode,
    ComparisonNode,
    ArithmeticNode,
    MembershipNode,
    LiteralNode,
    VariableRefNode,
    FunctionCallNode,
)


def simplify_drc(expr: DRCExpression) -> DRCExpression:
    """Simplify a DRC expression by applying transformations to its condition."""
    return DRCExpression(
        result_variables=expr.result_variables,
        condition=_simplify_condition(expr.condition),
    )


def _simplify_condition(node: DRCCondition) -> DRCCondition:
    """Recursively simplify a DRC condition."""
    if node is None:
        return node

    if isinstance(node, QuantifierNode):
        # First simplify the body
        simplified_body = _simplify_condition(node.body)

        # Merge nested quantifiers of the same kind:
        # ∃ X (∃ Y body) → ∃ X,Y body
        merged_vars = list(node.variables)
        current_body = simplified_body
        while isinstance(current_body, QuantifierNode) and current_body.kind == node.kind:
            merged_vars.extend(current_body.variables)
            current_body = _simplify_condition(current_body.body)

        return QuantifierNode(
            kind=node.kind,
            variables=merged_vars,
            body=current_body,
        )

    elif isinstance(node, LogicalConnectiveNode):
        return LogicalConnectiveNode(
            operator=node.operator,
            left=_simplify_condition(node.left),
            right=_simplify_condition(node.right),
        )

    elif isinstance(node, NotNode):
        return NotNode(operand=_simplify_condition(node.operand))

    elif isinstance(node, ComparisonNode):
        return ComparisonNode(
            operator=node.operator,
            left=_simplify_condition(node.left),
            right=_simplify_condition(node.right),
        )

    elif isinstance(node, ArithmeticNode):
        return ArithmeticNode(
            operator=node.operator,
            left=_simplify_condition(node.left),
            right=_simplify_condition(node.right),
        )

    elif isinstance(node, FunctionCallNode):
        return FunctionCallNode(
            function=node.function,
            arguments=[_simplify_condition(arg) for arg in node.arguments],
        )

    # Leaf nodes: MembershipNode, LiteralNode, VariableRefNode — no simplification
    return node
