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
    Infers variable types from usage (String if compared to string literal, Int otherwise).
    """
    relations: dict[str, int] = {}
    variables: set[str] = set()
    var_types: dict[str, str] = {}  # variable -> "Int" or "String"
    _collect_symbols(condition, relations, variables)
    _infer_types(condition, var_types)

    lines: list[str] = []
    lines.append("(set-logic ALL)")

    # Declare all variables with inferred types
    for var in sorted(variables):
        sort = var_types.get(var, "Int")
        lines.append(f"(declare-const {var} {sort})")

    # Declare relations as uninterpreted functions returning Bool
    # Use mixed sorts based on the variables used in membership
    rel_sorts: dict[str, list[str]] = {}
    _collect_relation_sorts(condition, var_types, rel_sorts)
    for rel_name, arity in sorted(relations.items()):
        if rel_name in rel_sorts:
            sorts = " ".join(rel_sorts[rel_name])
        else:
            sorts = " ".join(["Int"] * arity)
        lines.append(f"(declare-fun {rel_name} ({sorts}) Bool)")

    # Assert the formula
    formula = _convert_node(condition, var_types)
    lines.append(f"(assert {formula})")
    lines.append("(check-sat)")

    return "\n".join(lines)


def convert_condition_to_formula(condition: DRCCondition) -> str:
    """Convert a DRC condition to an SMT-LIB formula string (no script wrapper)."""
    var_types: dict[str, str] = {}
    _infer_types(condition, var_types)
    return _convert_node(condition, var_types)


def collect_symbols(condition: DRCCondition) -> tuple[dict[str, int], set[str]]:
    """Collect relation names (with arities) and variable names from a condition.

    Returns (relations_dict, variables_set).
    """
    relations: dict[str, int] = {}
    variables: set[str] = set()
    _collect_symbols(condition, relations, variables)
    return relations, variables


def collect_var_types(condition: DRCCondition) -> dict[str, str]:
    """Collect inferred variable types from a condition."""
    var_types: dict[str, str] = {}
    _infer_types(condition, var_types)
    return var_types


def _infer_types(node: DRCCondition, var_types: dict[str, str]) -> None:
    """Infer variable types from context (comparisons with literals)."""
    if node is None:
        return

    if isinstance(node, ComparisonNode):
        # If one side is a string literal and the other is a variable, mark it as String
        if isinstance(node.left, VariableRefNode) and isinstance(node.right, LiteralNode):
            if node.right.data_type == "string":
                var_types[node.left.name] = "String"
            elif node.right.data_type == "number":
                var_types.setdefault(node.left.name, "Int")
        elif isinstance(node.right, VariableRefNode) and isinstance(node.left, LiteralNode):
            if node.left.data_type == "string":
                var_types[node.right.name] = "String"
            elif node.left.data_type == "number":
                var_types.setdefault(node.right.name, "Int")
        _infer_types(node.left, var_types)
        _infer_types(node.right, var_types)

    elif isinstance(node, LogicalConnectiveNode):
        _infer_types(node.left, var_types)
        _infer_types(node.right, var_types)

    elif isinstance(node, NotNode):
        _infer_types(node.operand, var_types)

    elif isinstance(node, QuantifierNode):
        _infer_types(node.body, var_types)

    elif isinstance(node, ArithmeticNode):
        # Variables in arithmetic are Int
        if isinstance(node.left, VariableRefNode):
            var_types.setdefault(node.left.name, "Int")
        if isinstance(node.right, VariableRefNode):
            var_types.setdefault(node.right.name, "Int")
        _infer_types(node.left, var_types)
        _infer_types(node.right, var_types)


def _collect_relation_sorts(
    node: DRCCondition,
    var_types: dict[str, str],
    rel_sorts: dict[str, list[str]],
) -> None:
    """Collect the sort signature for each relation based on variable types at membership sites."""
    if node is None:
        return

    if isinstance(node, MembershipNode):
        sorts = [var_types.get(v, "Int") for v in node.variables]
        # Keep the longest (most complete) sort list for each relation
        if node.relation not in rel_sorts or len(sorts) > len(rel_sorts[node.relation]):
            rel_sorts[node.relation] = sorts

    elif isinstance(node, LogicalConnectiveNode):
        _collect_relation_sorts(node.left, var_types, rel_sorts)
        _collect_relation_sorts(node.right, var_types, rel_sorts)

    elif isinstance(node, NotNode):
        _collect_relation_sorts(node.operand, var_types, rel_sorts)

    elif isinstance(node, QuantifierNode):
        _collect_relation_sorts(node.body, var_types, rel_sorts)

    elif isinstance(node, ComparisonNode):
        _collect_relation_sorts(node.left, var_types, rel_sorts)
        _collect_relation_sorts(node.right, var_types, rel_sorts)


def _collect_symbols(
    node: DRCCondition,
    relations: dict[str, int],
    variables: set[str],
) -> None:
    """Recursively collect relation names (with arities) and variable names."""
    if isinstance(node, QuantifierNode):
        for v in node.variables:
            variables.add(v)
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


def _convert_node(node: DRCCondition, var_types: dict[str, str] | None = None) -> str:
    """Recursively convert a DRC condition node to SMT-LIB syntax."""
    if var_types is None:
        var_types = {}
    if isinstance(node, QuantifierNode):
        return _convert_quantifier(node, var_types)
    elif isinstance(node, LogicalConnectiveNode):
        return _convert_logical(node, var_types)
    elif isinstance(node, NotNode):
        return _convert_not(node, var_types)
    elif isinstance(node, ComparisonNode):
        return _convert_comparison(node, var_types)
    elif isinstance(node, MembershipNode):
        return _convert_membership(node)
    elif isinstance(node, ArithmeticNode):
        return _convert_arithmetic(node, var_types)
    elif isinstance(node, LiteralNode):
        return _convert_literal(node)
    elif isinstance(node, VariableRefNode):
        return node.name
    else:
        raise ValueError(f"Unknown DRC node type: {type(node)}")


def _convert_quantifier(node: QuantifierNode, var_types: dict[str, str]) -> str:
    quantifier = node.kind  # "forall" or "exists"
    bindings = " ".join(f"({v} {var_types.get(v, 'Int')})" for v in node.variables)
    body = _convert_node(node.body, var_types)
    return f"({quantifier} ({bindings}) {body})"


def _convert_logical(node: LogicalConnectiveNode, var_types: dict[str, str]) -> str:
    left = _convert_node(node.left, var_types)
    right = _convert_node(node.right, var_types)
    op_map = {"and": "and", "or": "or", "implies": "=>"}
    op = op_map[node.operator]
    return f"({op} {left} {right})"


def _convert_not(node: NotNode, var_types: dict[str, str]) -> str:
    operand = _convert_node(node.operand, var_types)
    return f"(not {operand})"


def _convert_comparison(node: ComparisonNode, var_types: dict[str, str]) -> str:
    left = _convert_node(node.left, var_types)
    right = _convert_node(node.right, var_types)
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


def _convert_arithmetic(node: ArithmeticNode, var_types: dict[str, str]) -> str:
    left = _convert_node(node.left, var_types)
    right = _convert_node(node.right, var_types)
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
