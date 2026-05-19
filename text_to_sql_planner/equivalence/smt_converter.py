"""Convert DRC conditions to SMT-LIB syntax for cvc5."""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ArithmeticNode,
    ComparisonNode,
    DRCCondition,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)


def convert_to_smt(condition: DRCCondition) -> str:
    """Convert a DRC condition into a complete SMT-LIB script.

    Generates declarations for all relations as uninterpreted functions,
    asserts the formula, and includes (check-sat).
    """
    relations: dict[str, int] = {}
    variables: set[str] = set()
    _collect_symbols(condition, relations, variables)

    lines: list[str] = []
    lines.append("(set-logic ALL)")

    # Declare all variables as Int constants
    for var in sorted(variables):
        lines.append(f"(declare-const {var} Int)")

    # Declare relations as uninterpreted functions returning Bool
    for rel_name, arity in sorted(relations.items()):
        sorts = " ".join(["Int"] * arity)
        lines.append(f"(declare-fun {rel_name} ({sorts}) Bool)")

    # Assert the formula
    formula = _convert_node(condition)
    lines.append(f"(assert {formula})")
    lines.append("(check-sat)")

    return "\n".join(lines)


def _collect_symbols(
    node: DRCCondition,
    relations: dict[str, int],
    variables: set[str],
) -> None:
    """Recursively collect relation names (with arities) and variable names."""
    if isinstance(node, QuantifierNode):
        for v in node.variables:
            variables.add(v)
        if node.relation:
            relations[node.relation] = max(
                relations.get(node.relation, 0), len(node.variables)
            )
        if node.body is not None:
            _collect_symbols(node.body, relations, variables)

    elif isinstance(node, LogicalConnectiveNode):
        if node.left is not None:
            _collect_symbols(node.left, relations, variables)
        if node.right is not None:
            _collect_symbols(node.right, relations, variables)

    elif isinstance(node, NotNode):
        if node.operand is not None:
            _collect_symbols(node.operand, relations, variables)

    elif isinstance(node, ComparisonNode):
        if node.left is not None:
            _collect_symbols(node.left, relations, variables)
        if node.right is not None:
            _collect_symbols(node.right, relations, variables)

    elif isinstance(node, MembershipNode):
        for v in node.variables:
            variables.add(v)
        if node.relation:
            relations[node.relation] = max(
                relations.get(node.relation, 0), len(node.variables)
            )

    elif isinstance(node, ArithmeticNode):
        if node.left is not None:
            _collect_symbols(node.left, relations, variables)
        if node.right is not None:
            _collect_symbols(node.right, relations, variables)

    elif isinstance(node, VariableRefNode):
        variables.add(node.name)

    elif isinstance(node, LiteralNode):
        pass  # No symbols to collect


def _convert_node(node: DRCCondition) -> str:
    """Recursively convert a DRC condition node to SMT-LIB syntax."""
    if isinstance(node, QuantifierNode):
        return _convert_quantifier(node)
    elif isinstance(node, LogicalConnectiveNode):
        return _convert_logical(node)
    elif isinstance(node, NotNode):
        return _convert_not(node)
    elif isinstance(node, ComparisonNode):
        return _convert_comparison(node)
    elif isinstance(node, MembershipNode):
        return _convert_membership(node)
    elif isinstance(node, ArithmeticNode):
        return _convert_arithmetic(node)
    elif isinstance(node, LiteralNode):
        return _convert_literal(node)
    elif isinstance(node, VariableRefNode):
        return node.name
    else:
        raise ValueError(f"Unknown DRC node type: {type(node)}")


def _convert_quantifier(node: QuantifierNode) -> str:
    quantifier = node.kind  # "forall" or "exists"
    bindings = " ".join(f"({v} Int)" for v in node.variables)
    body = _convert_node(node.body)
    return f"({quantifier} (({bindings})) {body})" if len(node.variables) == 1 else f"({quantifier} ({bindings}) {body})"


def _convert_logical(node: LogicalConnectiveNode) -> str:
    left = _convert_node(node.left)
    right = _convert_node(node.right)
    op_map = {"and": "and", "or": "or", "implies": "=>"}
    op = op_map[node.operator]
    return f"({op} {left} {right})"


def _convert_not(node: NotNode) -> str:
    operand = _convert_node(node.operand)
    return f"(not {operand})"


def _convert_comparison(node: ComparisonNode) -> str:
    left = _convert_node(node.left)
    right = _convert_node(node.right)
    op_map = {
        "=": "=",
        "!=": "distinct",
        "<": "<",
        ">": ">",
        "<=": "<=",
        ">=": ">=",
    }
    op = op_map[node.operator]
    return f"({op} {left} {right})"


def _convert_membership(node: MembershipNode) -> str:
    args = " ".join(node.variables)
    return f"({node.relation} {args})"


def _convert_arithmetic(node: ArithmeticNode) -> str:
    left = _convert_node(node.left)
    right = _convert_node(node.right)
    return f"({node.operator} {left} {right})"


def _convert_literal(node: LiteralNode) -> str:
    if node.data_type == "string":
        # SMT-LIB string literals
        escaped = str(node.value).replace('"', '""')
        return f'"{escaped}"'
    else:
        # Numeric literal
        value = node.value
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)
