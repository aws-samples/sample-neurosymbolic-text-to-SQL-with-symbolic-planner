"""Pretty printer for DRC AST nodes using Unicode notation.

Formats a DRCExpression into human-readable Unicode notation like:
    {x, y | ∃ a,b ∈ R (a = x ∧ b > 5)}

Uses all-or-nothing semantics: on error, no partial string is produced.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ArithmeticAggregateVariable,
    ArithmeticNode,
    ColumnVariable,
    ComparisonNode,
    ConditionalAggregateVariable,
    DRCCondition,
    DRCExpression,
    FunctionCallNode,
    IsNotNullNode,
    LimitExpression,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    OrderByExpression,
    QuantifierNode,
    QueryExpression,
    ResultVariable,
    ScalarLiteralVariable,
    SortCriterion,
    VariableRefNode,
)
from text_to_sql_planner.types.errors import PrintError
from text_to_sql_planner.printer.lisp_printer import PrintSuccess, PrintFailure, PrintResult


# Unicode symbols
_FORALL = "\u2200"  # ∀
_EXISTS = "\u2203"  # ∃
_AND = "\u2227"     # ∧
_OR = "\u2228"      # ∨
_NOT = "\u00AC"     # ¬
_IN = "\u2208"      # ∈
_IMPLIES = "\u2192" # →

# Operator precedence (higher number = tighter binding)
_PRECEDENCE = {
    "implies": 1,
    "or": 2,
    "and": 3,
    "not": 4,
}

_CONNECTIVE_SYMBOLS = {
    "and": _AND,
    "or": _OR,
    "implies": _IMPLIES,
}


class _PrintInternalError(Exception):
    """Internal exception used to abort printing and propagate error info."""

    def __init__(self, print_error: PrintError):
        self.print_error = print_error
        super().__init__(print_error.message)


def pretty_print(expr: DRCExpression) -> PrintResult:
    """Format a DRCExpression as human-readable Unicode notation.

    Returns PrintSuccess with the formatted string on success,
    or PrintFailure with a PrintError on any validation failure.
    Guarantees all-or-nothing output.
    """
    if expr is None:
        return PrintFailure(error=PrintError(message="Input expression is None", node=None))

    if not isinstance(expr, DRCExpression):
        return PrintFailure(
            error=PrintError(message=f"Expected DRCExpression, got {type(expr).__name__}", node=expr)
        )

    try:
        result_vars_str = _print_result_variables(expr.result_variables)
        condition_str = _print_condition(expr.condition, parent_precedence=0)
        output = f"{{{result_vars_str} | {condition_str}}}"
        return PrintSuccess(output=output)
    except _PrintInternalError as e:
        return PrintFailure(error=e.print_error)


def pretty_print_query(query: QueryExpression) -> PrintResult:
    """Pretty-print a query (DRC, optionally wrapped in order-by/limit).

    Output examples::

        {emp_id, name, salary | (emp_id, name, salary) ∈ Employees}
        ORDER BY salary DESC {emp_id, name, salary | ...}
        LIMIT 5 ORDER BY salary DESC {emp_id, name, salary | ...}
    """
    if query is None:
        return PrintFailure(error=PrintError(message="Input query is None", node=None))

    try:
        return PrintSuccess(output=_pretty_query_node(query))
    except _PrintInternalError as e:
        return PrintFailure(error=e.print_error)


def _pretty_query_node(query: QueryExpression) -> str:
    if isinstance(query, DRCExpression):
        result = pretty_print(query)
        if isinstance(result, PrintFailure):
            raise _PrintInternalError(result.error)
        return result.output
    if isinstance(query, OrderByExpression):
        if not query.criteria:
            raise _PrintInternalError(
                PrintError(message="OrderByExpression has no criteria", node=query)
            )
        crit_str = ", ".join(_pretty_sort_criterion(c) for c in query.criteria)
        inner_str = _pretty_query_node(query.inner)
        return f"ORDER BY {crit_str} {inner_str}"
    if isinstance(query, LimitExpression):
        if query.n is None or query.n <= 0:
            raise _PrintInternalError(
                PrintError(message=f"LimitExpression n must be positive, got {query.n}", node=query)
            )
        inner_str = _pretty_query_node(query.inner)
        return f"LIMIT {query.n} {inner_str}"
    raise _PrintInternalError(
        PrintError(message=f"Unknown query node type: {type(query).__name__}", node=query)
    )


def _pretty_sort_criterion(c: SortCriterion) -> str:
    if c is None:
        raise _PrintInternalError(PrintError(message="SortCriterion is None", node=None))
    if not c.column:
        raise _PrintInternalError(PrintError(message="SortCriterion has empty column", node=c))
    if c.direction not in ("asc", "desc"):
        raise _PrintInternalError(
            PrintError(message=f"Invalid sort direction: {c.direction}", node=c)
        )
    head = f"{c.aggregate}({c.column})" if c.aggregate else c.column
    return f"{head} {c.direction.upper()}"


def _print_agg_operand_pretty(operand) -> str:
    """Print an aggregate operand in pretty form."""
    if isinstance(operand, ScalarLiteralVariable):
        return str(operand.value)
    if isinstance(operand, ColumnVariable):
        return operand.name
    if isinstance(operand, ConditionalAggregateVariable):
        cond_str = _print_condition(operand.condition)
        return f"{operand.function}_IF({cond_str}, {operand.column})"
    if isinstance(operand, ArithmeticAggregateVariable):
        left_str = _print_agg_operand_pretty(operand.left)
        right_str = _print_agg_operand_pretty(operand.right)
        return f"{left_str} {operand.operator} {right_str}"
    return f"{operand.function}({operand.column})"


def _print_result_variables(variables: list[ResultVariable]) -> str:
    """Print the result variables as comma-separated list.

    Plain columns come first, then aggregates.
    """
    if variables is None:
        raise _PrintInternalError(PrintError(message="Result variables list is None", node=None))

    plain_parts: list[str] = []
    agg_parts: list[str] = []
    for var in variables:
        if var is None:
            raise _PrintInternalError(PrintError(message="Result variable is None", node=None))
        if isinstance(var, ColumnVariable):
            if not var.name:
                raise _PrintInternalError(
                    PrintError(message="ColumnVariable has empty name", node=var)
                )
            plain_parts.append(var.name)
        elif isinstance(var, AggregateVariable):
            if not var.function:
                raise _PrintInternalError(
                    PrintError(message="AggregateVariable has empty function", node=var)
                )
            if not var.column:
                raise _PrintInternalError(
                    PrintError(message="AggregateVariable has empty column", node=var)
                )
            agg_parts.append(f"{var.function}({var.column})")
        elif isinstance(var, ArithmeticAggregateVariable):
            left_str = _print_agg_operand_pretty(var.left)
            right_str = _print_agg_operand_pretty(var.right)
            agg_parts.append(f"{left_str} {var.operator} {right_str}")
        elif isinstance(var, ConditionalAggregateVariable):
            cond_str = _print_condition(var.condition)
            agg_parts.append(f"{var.function}_IF({cond_str}, {var.column})")
        elif isinstance(var, ScalarLiteralVariable):
            agg_parts.append(str(var.value))
        else:
            raise _PrintInternalError(
                PrintError(message=f"Unknown result variable type: {type(var).__name__}", node=var)
            )
    return ", ".join(plain_parts + agg_parts)


def _print_condition(node: DRCCondition, parent_precedence: int = 0) -> str:
    """Print a DRC condition node recursively with precedence-based parenthesization."""
    if node is None:
        raise _PrintInternalError(PrintError(message="Condition node is None", node=None))

    if isinstance(node, VariableRefNode):
        if not node.name:
            raise _PrintInternalError(
                PrintError(message="VariableRefNode has empty name", node=node)
            )
        return node.name

    if isinstance(node, LiteralNode):
        if node.data_type == "string":
            escaped = str(node.value).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        else:
            return str(node.value)

    if isinstance(node, MembershipNode):
        if not node.relation:
            raise _PrintInternalError(
                PrintError(message="MembershipNode has empty relation", node=node)
            )
        if node.variables is None:
            raise _PrintInternalError(
                PrintError(message="MembershipNode has None variables", node=node)
            )
        vars_str = ",".join(node.variables)
        return f"{vars_str} {_IN} {node.relation}"

    if isinstance(node, IsNotNullNode):
        if not node.column:
            raise _PrintInternalError(
                PrintError(message="IsNotNullNode has empty column", node=node)
            )
        return f"{node.column} IS NOT NULL"

    if isinstance(node, QuantifierNode):
        if node.kind not in ("forall", "exists"):
            raise _PrintInternalError(
                PrintError(message=f"Invalid quantifier kind: {node.kind}", node=node)
            )
        if node.variables is None:
            raise _PrintInternalError(
                PrintError(message="QuantifierNode has None variables", node=node)
            )
        symbol = _EXISTS if node.kind == "exists" else _FORALL
        vars_str = ",".join(node.variables)
        body_str = _print_condition(node.body, parent_precedence=0)
        return f"{symbol} {vars_str} ({body_str})"

    if isinstance(node, NotNode):
        operand_str = _print_condition(node.operand, parent_precedence=_PRECEDENCE["not"])
        return f"{_NOT}{operand_str}"

    if isinstance(node, LogicalConnectiveNode):
        if node.operator not in ("and", "or", "implies"):
            raise _PrintInternalError(
                PrintError(message=f"Invalid logical operator: {node.operator}", node=node)
            )
        my_precedence = _PRECEDENCE[node.operator]
        symbol = _CONNECTIVE_SYMBOLS[node.operator]

        left_str = _print_condition(node.left, parent_precedence=0)
        right_str = _print_condition(node.right, parent_precedence=0)

        # Parenthesize children if they have strictly lower precedence
        if _needs_parens(node.left, my_precedence):
            left_str = f"({left_str})"
        if _needs_parens(node.right, my_precedence):
            right_str = f"({right_str})"

        result = f"{left_str} {symbol} {right_str}"

        # Parenthesize ourselves if parent has higher precedence
        if parent_precedence > my_precedence:
            result = f"({result})"
        return result

    if isinstance(node, ComparisonNode):
        if node.operator not in ("=", "!=", "<", ">", "<=", ">="):
            raise _PrintInternalError(
                PrintError(message=f"Invalid comparison operator: {node.operator}", node=node)
            )
        left_str = _print_condition(node.left, parent_precedence=0)
        right_str = _print_condition(node.right, parent_precedence=0)
        return f"{left_str} {node.operator} {right_str}"

    if isinstance(node, ArithmeticNode):
        if node.operator not in ("+", "-", "*", "/"):
            raise _PrintInternalError(
                PrintError(message=f"Invalid arithmetic operator: {node.operator}", node=node)
            )
        left_str = _print_condition(node.left, parent_precedence=0)
        right_str = _print_condition(node.right, parent_precedence=0)
        # Parenthesize arithmetic sub-expressions if they are also arithmetic with lower precedence
        if _needs_arith_parens(node.left, node.operator):
            left_str = f"({left_str})"
        if _needs_arith_parens(node.right, node.operator):
            right_str = f"({right_str})"
        return f"{left_str} {node.operator} {right_str}"

    if isinstance(node, FunctionCallNode):
        if not node.arguments:
            return node.function  # No-arg constant like CURRENT_DATE
        args_str = ", ".join(_print_condition(arg, parent_precedence=0) for arg in node.arguments)
        return f"{node.function}({args_str})"

    raise _PrintInternalError(
        PrintError(message=f"Unknown condition node type: {type(node).__name__}", node=node)
    )


def _needs_parens(child: DRCCondition, parent_prec: int) -> bool:
    """Check if a child logical connective needs parentheses given parent precedence."""
    if isinstance(child, LogicalConnectiveNode):
        child_prec = _PRECEDENCE.get(child.operator, 0)
        return child_prec < parent_prec
    return False


def _is_redundant_body(node: QuantifierNode) -> bool:
    """No longer used — quantifiers don't have a relation field anymore."""
    return False


def _needs_arith_parens(child: DRCCondition, parent_op: str) -> bool:
    """Check if an arithmetic child needs parentheses.

    Multiplicative operators (* /) bind tighter than additive (+ -).
    """
    if not isinstance(child, ArithmeticNode):
        return False
    # * and / have higher precedence than + and -
    high_prec = {"*", "/"}
    low_prec = {"+", "-"}
    if parent_op in high_prec and child.operator in low_prec:
        return True
    return False


def pretty_print_indented(expr: DRCExpression, width: int = 80) -> PrintResult:
    """Format a DRCExpression with intelligent indentation for long expressions.

    If the single-line output exceeds `width` characters, produces a
    multi-line indented version. Otherwise returns the single-line form.
    Sub-expressions that fit within `width` stay on one line.
    """
    # First try the compact single-line version
    result = pretty_print(expr)
    if not isinstance(result, PrintSuccess):
        return result

    if len(result.output) <= width:
        return result

    # Need indented version
    if expr is None:
        return PrintFailure(error=PrintError(message="Input expression is None", node=None))

    try:
        result_vars_str = _print_result_variables(expr.result_variables)
        condition_str = _print_condition_indented(expr.condition, indent=2, width=width, parent_precedence=0)
        output = f"{{{result_vars_str} |\n{condition_str}\n}}"
        return PrintSuccess(output=output)
    except _PrintInternalError as e:
        return PrintFailure(error=e.print_error)


def _print_condition_indented(node: DRCCondition, indent: int = 0, width: int = 80, parent_precedence: int = 0) -> str:
    """Print a DRC condition with indentation for nested structures.

    If a sub-expression fits on one line within `width - indent`, use the
    compact single-line form. Only break into multiple lines when needed.
    """
    pad = " " * indent

    if node is None:
        raise _PrintInternalError(PrintError(message="Condition node is None", node=None))

    # Try compact form first — if it fits, use it
    compact = _print_condition(node, parent_precedence=parent_precedence)
    if len(compact) + indent <= width:
        return f"{pad}{compact}"

    # Doesn't fit — break it up based on node type

    if isinstance(node, VariableRefNode):
        return f"{pad}{compact}"

    if isinstance(node, LiteralNode):
        return f"{pad}{compact}"

    if isinstance(node, MembershipNode):
        return f"{pad}{compact}"

    if isinstance(node, QuantifierNode):
        if node.kind not in ("forall", "exists"):
            raise _PrintInternalError(PrintError(message=f"Invalid quantifier kind: {node.kind}", node=node))
        symbol = _EXISTS if node.kind == "exists" else _FORALL
        vars_str = ",".join(node.variables)
        header = f"{symbol} {vars_str} ("
        body_str = _print_condition_indented(node.body, indent=indent + 2, width=width, parent_precedence=0)
        # Check if the whole thing fits on one line
        body_compact = _print_condition(node.body, parent_precedence=0)
        one_line = f"{header}{body_compact})"
        if len(one_line) + indent <= width:
            return f"{pad}{one_line}"
        return f"{pad}{header}\n{body_str}\n{pad})"

    if isinstance(node, NotNode):
        operand_str = _print_condition_indented(node.operand, indent=indent + 2, width=width, parent_precedence=_PRECEDENCE["not"])
        return f"{pad}{_NOT}(\n{operand_str}\n{pad})"

    if isinstance(node, LogicalConnectiveNode):
        if node.operator not in ("and", "or", "implies"):
            raise _PrintInternalError(PrintError(message=f"Invalid logical operator: {node.operator}", node=node))
        symbol = _CONNECTIVE_SYMBOLS[node.operator]
        my_precedence = _PRECEDENCE[node.operator]

        # Flatten chains of the same operator (a ∧ b ∧ c ∧ d)
        operands = _flatten_connective(node, node.operator)

        # Render each operand compactly and pair with the AST node
        rendered: list[tuple[str, DRCCondition]] = []
        for operand in operands:
            operand_compact = _print_condition(operand, parent_precedence=my_precedence)
            if _needs_parens(operand, my_precedence):
                operand_compact = f"({operand_compact})"
            rendered.append((operand_compact, operand))

        # Group operands onto lines, joining with the symbol
        # Each line should be ≤ width characters (including indent)
        # When an operand is complex (multi-line), give it its own line with ∧ prefix
        lines: list[str] = []
        current_parts: list[str] = []
        current_len = indent

        for item_text, item_node in rendered:
            is_complex = isinstance(item_node, (QuantifierNode, LogicalConnectiveNode))
            joiner = f" {symbol} "
            joiner_len = len(joiner) if current_parts else 0
            needed = joiner_len + len(item_text)

            # Complex operands always get their own line
            if is_complex and len(item_text) + indent > width:
                # Flush any accumulated simple parts first
                if current_parts:
                    lines.append(pad + f" {symbol} ".join(current_parts))
                    current_parts = []
                    current_len = indent
                # Expand this operand with indentation
                expanded = _print_condition_indented(item_node, indent=indent, width=width, parent_precedence=my_precedence)
                if lines:
                    lines.append(f"{pad}{symbol} {expanded.lstrip()}")
                else:
                    lines.append(expanded)
                continue

            if current_parts and current_len + needed > width:
                # Flush current line
                lines.append(pad + f" {symbol} ".join(current_parts))
                current_parts = []
                current_len = indent

            current_parts.append(item_text)
            current_len = indent + len(f" {symbol} ".join(current_parts))

        if current_parts:
            if lines:
                lines.append(f"{pad}{symbol} " + f" {symbol} ".join(current_parts))
            else:
                lines.append(pad + f" {symbol} ".join(current_parts))

        return "\n".join(lines)

    if isinstance(node, ComparisonNode):
        return f"{pad}{compact}"

    if isinstance(node, ArithmeticNode):
        return f"{pad}{compact}"

    raise _PrintInternalError(
        PrintError(message=f"Unknown condition node type: {type(node).__name__}", node=node)
    )


def _flatten_connective(node: DRCCondition, operator: str) -> list[DRCCondition]:
    """Flatten a chain of the same logical connective into a list of operands.

    e.g. (and (and a b) c) → [a, b, c]
    """
    if isinstance(node, LogicalConnectiveNode) and node.operator == operator:
        return _flatten_connective(node.left, operator) + _flatten_connective(node.right, operator)
    return [node]
