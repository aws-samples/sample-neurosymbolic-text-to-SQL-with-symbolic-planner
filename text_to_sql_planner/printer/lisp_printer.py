"""Lisp S-expression printer for DRC AST nodes.

Serializes a DRCExpression back to Lisp S-expression format with
all-or-nothing semantics: on error, no partial string is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

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


@dataclass
class PrintSuccess:
    output: str


@dataclass
class PrintFailure:
    error: PrintError


PrintResult = Union[PrintSuccess, PrintFailure]


def print_lisp(expr: DRCExpression) -> PrintResult:
    """Serialize a DRCExpression to Lisp S-expression format.

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
        condition_str = _print_condition(expr.condition)
        output = f"(drc ({result_vars_str}) {condition_str})"
        return PrintSuccess(output=output)
    except _PrintInternalError as e:
        return PrintFailure(error=e.print_error)


class _PrintInternalError(Exception):
    """Internal exception used to abort printing and propagate error info."""

    def __init__(self, print_error: PrintError):
        self.print_error = print_error
        super().__init__(print_error.message)


def _print_result_variables(variables: list[ResultVariable]) -> str:
    """Print the result variables list."""
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
            parts.append(f"({var.function} {var.column})")
        else:
            raise _PrintInternalError(
                PrintError(message=f"Unknown result variable type: {type(var).__name__}", node=var)
            )
    return " ".join(parts)


def _print_condition(node: DRCCondition) -> str:
    """Print a DRC condition node recursively."""
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
        vars_str = " ".join(node.variables)
        return f"(in ({vars_str}) {node.relation})"

    if isinstance(node, QuantifierNode):
        if node.kind not in ("forall", "exists"):
            raise _PrintInternalError(
                PrintError(message=f"Invalid quantifier kind: {node.kind}", node=node)
            )
        if node.variables is None:
            raise _PrintInternalError(
                PrintError(message="QuantifierNode has None variables", node=node)
            )
        vars_str = " ".join(node.variables)
        body_str = _print_condition(node.body)
        return f"({node.kind} ({vars_str}) {body_str})"

    if isinstance(node, LogicalConnectiveNode):
        if node.operator not in ("and", "or", "implies"):
            raise _PrintInternalError(
                PrintError(message=f"Invalid logical operator: {node.operator}", node=node)
            )
        left_str = _print_condition(node.left)
        right_str = _print_condition(node.right)
        return f"({node.operator} {left_str} {right_str})"

    if isinstance(node, NotNode):
        operand_str = _print_condition(node.operand)
        return f"(not {operand_str})"

    if isinstance(node, ComparisonNode):
        if node.operator not in ("=", "!=", "<", ">", "<=", ">="):
            raise _PrintInternalError(
                PrintError(message=f"Invalid comparison operator: {node.operator}", node=node)
            )
        left_str = _print_condition(node.left)
        right_str = _print_condition(node.right)
        return f"({node.operator} {left_str} {right_str})"

    if isinstance(node, ArithmeticNode):
        if node.operator not in ("+", "-", "*", "/"):
            raise _PrintInternalError(
                PrintError(message=f"Invalid arithmetic operator: {node.operator}", node=node)
            )
        left_str = _print_condition(node.left)
        right_str = _print_condition(node.right)
        return f"({node.operator} {left_str} {right_str})"

    raise _PrintInternalError(
        PrintError(message=f"Unknown condition node type: {type(node).__name__}", node=node)
    )
