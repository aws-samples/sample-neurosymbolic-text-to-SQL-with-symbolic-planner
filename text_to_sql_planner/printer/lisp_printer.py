"""Lisp S-expression printer for DRC AST nodes.

Serializes a DRCExpression back to Lisp S-expression format with
all-or-nothing semantics: on error, no partial string is produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ArithmeticNode,
    ColumnVariable,
    ComparisonNode,
    DRCCondition,
    DRCExpression,
    FunctionCallNode,
    LimitExpression,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    OrderByExpression,
    QuantifierNode,
    QueryExpression,
    ResultVariable,
    SortCriterion,
    VariableRefNode,
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


def print_query_lisp(query: QueryExpression) -> PrintResult:
    """Serialize a query (DRC, optionally wrapped in order-by/limit) to
    Lisp S-expression format.

    Examples of output::

        (drc (...) ...)
        (order-by ((salary desc)) (drc (...) ...))
        (limit 5 (order-by ((salary desc)) (drc (...) ...)))
    """
    if query is None:
        return PrintFailure(error=PrintError(message="Input query is None", node=None))

    try:
        return PrintSuccess(output=_print_query_node(query))
    except _PrintInternalError as e:
        return PrintFailure(error=e.print_error)


def _print_query_node(query: QueryExpression) -> str:
    if isinstance(query, DRCExpression):
        result = print_lisp(query)
        if isinstance(result, PrintFailure):
            raise _PrintInternalError(result.error)
        return result.output
    if isinstance(query, OrderByExpression):
        return _print_order_by(query)
    if isinstance(query, LimitExpression):
        return _print_limit(query)
    raise _PrintInternalError(
        PrintError(message=f"Unknown query node type: {type(query).__name__}", node=query)
    )


def _print_sort_criterion(c: SortCriterion) -> str:
    if c is None:
        raise _PrintInternalError(PrintError(message="SortCriterion is None", node=None))
    if not c.column:
        raise _PrintInternalError(
            PrintError(message="SortCriterion has empty column", node=c)
        )
    if c.direction not in ("asc", "desc"):
        raise _PrintInternalError(
            PrintError(message=f"Invalid sort direction: {c.direction}", node=c)
        )
    if c.aggregate is not None:
        if c.aggregate not in ("COUNT", "SUM", "AVG", "MIN", "MAX"):
            raise _PrintInternalError(
                PrintError(message=f"Invalid aggregate in sort key: {c.aggregate}", node=c)
            )
        return f"(({c.aggregate} {c.column}) {c.direction})"
    return f"({c.column} {c.direction})"


def _print_order_by(node: OrderByExpression) -> str:
    if not node.criteria:
        raise _PrintInternalError(
            PrintError(message="OrderByExpression has no criteria", node=node)
        )
    if node.inner is None:
        raise _PrintInternalError(
            PrintError(message="OrderByExpression has no inner query", node=node)
        )
    crit_str = " ".join(_print_sort_criterion(c) for c in node.criteria)
    inner_str = _print_query_node(node.inner)
    return f"(order-by ({crit_str}) {inner_str})"


def _print_limit(node: LimitExpression) -> str:
    if node.n is None or node.n <= 0:
        raise _PrintInternalError(
            PrintError(message=f"LimitExpression n must be a positive int, got {node.n}", node=node)
        )
    if node.inner is None:
        raise _PrintInternalError(
            PrintError(message="LimitExpression has no inner query", node=node)
        )
    inner_str = _print_query_node(node.inner)
    return f"(limit {node.n} {inner_str})"


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

    if isinstance(node, FunctionCallNode):
        if not node.function:
            raise _PrintInternalError(
                PrintError(message="FunctionCallNode has empty function name", node=node)
            )
        if not node.arguments:
            return node.function  # No-arg constant like CURRENT_DATE
        args_str = " ".join(_print_condition(arg) for arg in node.arguments)
        return f"({node.function} {args_str})"

    raise _PrintInternalError(
        PrintError(message=f"Unknown condition node type: {type(node).__name__}", node=node)
    )
