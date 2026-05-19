"""Pretty printer for DRC AST nodes using Unicode notation.

Formats a DRCExpression into human-readable Unicode notation like:
    {x, y | ∃ a,b ∈ R (a = x ∧ b > 5)}

Uses all-or-nothing semantics: on error, no partial string is produced.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    DRCExpression,
    ColumnVariable,
    AggregateVariable,
    QuantifierNode,
    LogicalConnectiveNode,
    NotNode,
    ComparisonNode,
    MembershipNode,
    ArithmeticNode,
    LiteralNode,
    VariableRefNode,
    DRCCondition,
    ResultVariable,
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


def _print_result_variables(variables: list[ResultVariable]) -> str:
    """Print the result variables as comma-separated list."""
    if variables is None:
        raise _PrintInternalError(PrintError(message="Result variables list is None", node=None))

    parts: list[str] = []
    for var in variables:
        if var is None:
            raise _PrintInternalError(PrintError(message="Result variable is None", node=None))
        if isinstance(var, ColumnVariable):
            if not var.name:
                raise _PrintInternalError(
                    PrintError(message="ColumnVariable has empty name", node=var)
                )
            parts.append(var.name)
        elif isinstance(var, AggregateVariable):
            if not var.function:
                raise _PrintInternalError(
                    PrintError(message="AggregateVariable has empty function", node=var)
                )
            if not var.column:
                raise _PrintInternalError(
                    PrintError(message="AggregateVariable has empty column", node=var)
                )
            parts.append(f"{var.function}({var.column})")
        else:
            raise _PrintInternalError(
                PrintError(message=f"Unknown result variable type: {type(var).__name__}", node=var)
            )
    return ", ".join(parts)


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

    if isinstance(node, QuantifierNode):
        if node.kind not in ("forall", "exists"):
            raise _PrintInternalError(
                PrintError(message=f"Invalid quantifier kind: {node.kind}", node=node)
            )
        if not node.relation:
            raise _PrintInternalError(
                PrintError(message="QuantifierNode has empty relation", node=node)
            )
        if node.variables is None:
            raise _PrintInternalError(
                PrintError(message="QuantifierNode has None variables", node=node)
            )
        symbol = _EXISTS if node.kind == "exists" else _FORALL
        vars_str = ",".join(node.variables)
        body_str = _print_condition(node.body, parent_precedence=0)
        return f"{symbol} {vars_str} {_IN} {node.relation} ({body_str})"

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

    raise _PrintInternalError(
        PrintError(message=f"Unknown condition node type: {type(node).__name__}", node=node)
    )


def _needs_parens(child: DRCCondition, parent_prec: int) -> bool:
    """Check if a child logical connective needs parentheses given parent precedence."""
    if isinstance(child, LogicalConnectiveNode):
        child_prec = _PRECEDENCE.get(child.operator, 0)
        return child_prec < parent_prec
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
