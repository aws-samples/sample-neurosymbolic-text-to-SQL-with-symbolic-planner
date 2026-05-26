"""Convert DRC conditions to SMT-LIB syntax for cvc5."""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ArithmeticNode,
    ComparisonNode,
    DRCCondition,
    FunctionCallNode,
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
    """Infer variable types from context (comparisons with literals).

    Rules:
    - Variable compared with = or != to a string literal → String (always wins)
    - Variable compared with <, >, <=, >= to a string literal → Int
      (dates/ordered strings are modeled as Int since SMT-LIB String doesn't support ordering)
    - Variable compared to a number → Int (only if not already String)
    - Variable used in arithmetic → Int (only if not already String)
    - String type always wins over Int (once a variable is seen as String, it stays String)
    """
    if node is None:
        return

    if isinstance(node, ComparisonNode):
        # Ordering comparisons force Int even for string literals (dates, etc.)
        is_ordering = node.operator in ("<", ">", "<=", ">=")

        if isinstance(node.left, VariableRefNode) and isinstance(node.right, LiteralNode):
            name = node.left.name
            if node.right.data_type == "string":
                if is_ordering:
                    # Dates and ordered strings → model as Int, but only if not already String
                    if var_types.get(name) != "String":
                        var_types[name] = "Int"
                else:
                    # String equality → always String (wins over Int)
                    var_types[name] = "String"
            elif node.right.data_type == "number":
                if var_types.get(name) != "String":
                    var_types.setdefault(name, "Int")
        elif isinstance(node.right, VariableRefNode) and isinstance(node.left, LiteralNode):
            name = node.right.name
            if node.left.data_type == "string":
                if is_ordering:
                    if var_types.get(name) != "String":
                        var_types[name] = "Int"
                else:
                    var_types[name] = "String"
            elif node.left.data_type == "number":
                if var_types.get(name) != "String":
                    var_types.setdefault(name, "Int")
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

    elif isinstance(node, FunctionCallNode):
        for arg in node.arguments:
            if arg is not None:
                _collect_symbols(arg, relations, variables)


def _convert_node(node: DRCCondition, var_types: dict[str, str] | None = None, scope: dict[str, str] | None = None) -> str:
    """Recursively convert a DRC condition node to SMT-LIB syntax.
    
    `scope` maps original variable names to their (possibly renamed) SMT-LIB names,
    handling shadowing in nested quantifiers.
    """
    if var_types is None:
        var_types = {}
    if scope is None:
        scope = {}
    if isinstance(node, QuantifierNode):
        return _convert_quantifier(node, var_types, scope)
    elif isinstance(node, LogicalConnectiveNode):
        return _convert_logical(node, var_types, scope)
    elif isinstance(node, NotNode):
        return _convert_not(node, var_types, scope)
    elif isinstance(node, ComparisonNode):
        return _convert_comparison(node, var_types, scope)
    elif isinstance(node, MembershipNode):
        return _convert_membership(node, scope)
    elif isinstance(node, ArithmeticNode):
        return _convert_arithmetic(node, var_types, scope)
    elif isinstance(node, LiteralNode):
        return _convert_literal(node)
    elif isinstance(node, FunctionCallNode):
        return _convert_function_call(node, var_types, scope)
    elif isinstance(node, VariableRefNode):
        # Use the renamed name if in scope
        return scope.get(node.name, node.name)
    else:
        raise ValueError(f"Unknown DRC node type: {type(node)}")


_rename_counter: dict[str, int] = {}


def _fresh_name(base: str, all_names: set[str]) -> str:
    """Generate a fresh variable name that doesn't conflict with existing names."""
    if base not in all_names:
        return base
    counter = 2
    while f"{base}_{counter}" in all_names:
        counter += 1
    return f"{base}_{counter}"


def _convert_quantifier(node: QuantifierNode, var_types: dict[str, str], scope: dict[str, str]) -> str:
    quantifier = node.kind  # "forall" or "exists"
    
    # Alpha-rename variables that shadow outer scope
    all_in_scope = set(scope.values())
    new_scope = dict(scope)  # copy outer scope
    renamed_vars: list[str] = []
    
    for v in node.variables:
        if v in all_in_scope:
            # This variable shadows an outer one — rename it
            fresh = _fresh_name(v, all_in_scope)
            new_scope[v] = fresh
            all_in_scope.add(fresh)
            renamed_vars.append(fresh)
        else:
            new_scope[v] = v
            all_in_scope.add(v)
            renamed_vars.append(v)
    
    bindings = " ".join(f"({rv} {var_types.get(orig, 'Int')})" 
                        for rv, orig in zip(renamed_vars, node.variables))
    body = _convert_node(node.body, var_types, new_scope)
    return f"({quantifier} ({bindings}) {body})"


def _convert_logical(node: LogicalConnectiveNode, var_types: dict[str, str], scope: dict[str, str]) -> str:
    left = _convert_node(node.left, var_types, scope)
    right = _convert_node(node.right, var_types, scope)
    op_map = {"and": "and", "or": "or", "implies": "=>"}
    op = op_map[node.operator]
    return f"({op} {left} {right})"


def _convert_not(node: NotNode, var_types: dict[str, str], scope: dict[str, str]) -> str:
    operand = _convert_node(node.operand, var_types, scope)
    return f"(not {operand})"


def _convert_comparison(node: ComparisonNode, var_types: dict[str, str], scope: dict[str, str]) -> str:
    is_ordering = node.operator in ("<", ">", "<=", ">=")

    left = _convert_node_for_comparison(node.left, var_types, is_ordering, scope)
    right = _convert_node_for_comparison(node.right, var_types, is_ordering, scope)

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


def _convert_node_for_comparison(node: DRCCondition, var_types: dict[str, str], is_ordering: bool, scope: dict[str, str] | None = None) -> str:
    """Convert a node for use in a comparison, handling date literals."""
    if is_ordering and isinstance(node, LiteralNode) and node.data_type == "string":
        int_val = _date_string_to_int(str(node.value))
        if int_val is not None:
            return str(int_val)
    return _convert_node(node, var_types, scope)


def _date_string_to_int(s: str) -> int | None:
    """Convert a date string like '2000-01-01' to days since epoch (1970-01-01).

    Returns None if the string doesn't look like a date.
    """
    import re
    from datetime import date

    # Match YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})$", s)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            epoch = date(1970, 1, 1)
            return (d - epoch).days
        except ValueError:
            return None
    return None


def _convert_membership(node: MembershipNode, scope: dict[str, str] | None = None) -> str:
    if scope is None:
        scope = {}
    args = " ".join(scope.get(v, v) for v in node.variables)
    return f"({node.relation} {args})"


def _convert_arithmetic(node: ArithmeticNode, var_types: dict[str, str], scope: dict[str, str] | None = None) -> str:
    left = _convert_node(node.left, var_types, scope)
    right = _convert_node(node.right, var_types, scope)
    return f"({node.operator} {left} {right})"


def _convert_function_call(node: FunctionCallNode, var_types: dict[str, str], scope: dict[str, str] | None = None) -> str:
    """Convert a function call to SMT-LIB.

    CURRENT_DATE → today's epoch days (integer)
    DATE_SUB(x, n) → (- x n)
    DATE_ADD(x, n) → (+ x n)
    DATEDIFF(x, y) → (- x y)
    YEAR(x), MONTH(x), DAY(x) → treated as uninterpreted functions
    """
    from datetime import date

    if node.function == "CURRENT_DATE":
        today = date.today()
        epoch = date(1970, 1, 1)
        days = (today - epoch).days
        return str(days)
    elif node.function == "DATE_SUB" and len(node.arguments) == 2:
        left = _convert_node(node.arguments[0], var_types, scope)
        right = _convert_node(node.arguments[1], var_types, scope)
        return f"(- {left} {right})"
    elif node.function == "DATE_ADD" and len(node.arguments) == 2:
        left = _convert_node(node.arguments[0], var_types, scope)
        right = _convert_node(node.arguments[1], var_types, scope)
        return f"(+ {left} {right})"
    elif node.function == "DATEDIFF" and len(node.arguments) == 2:
        left = _convert_node(node.arguments[0], var_types, scope)
        right = _convert_node(node.arguments[1], var_types, scope)
        return f"(- {left} {right})"
    else:
        # Generic uninterpreted function
        if not node.arguments:
            return node.function
        args = " ".join(_convert_node(arg, var_types, scope) for arg in node.arguments)
        return f"({node.function} {args})"


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
