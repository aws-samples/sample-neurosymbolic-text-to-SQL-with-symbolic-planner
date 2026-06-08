"""Internal expression-level AST for the SQL→DRC converter.

This AST is the parser's intermediate representation; it is not part of the
public ``bird_benchmark`` API. Nodes carry 1-indexed source coordinates via
:class:`Position` so the parser/translator can attach line and column
information to :class:`bird_benchmark.types.ConverterError` instances.

The shapes here match the "Components and Interfaces / Expression-level AST"
section of the design document one-for-one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal as _Lit
from typing import Union


# --- Source position -------------------------------------------------------


@dataclass
class Position:
    """1-indexed (line, column) coordinate into the original SQL source."""

    line: int
    column: int


# --- Aggregate function names ---------------------------------------------

AggregateFunction = _Lit["COUNT", "SUM", "AVG", "MIN", "MAX"]


# --- Expression nodes ------------------------------------------------------


@dataclass
class ColumnRef:
    qualifier: str | None
    name: str
    pos: Position
    # ``quoted`` is True when the column name (the unqualified leaf token)
    # came from a SQLite-style double-quoted token like ``"name"``. The
    # translator applies SQLite's identifier-fallback behaviour (Req 13.2 /
    # 13.3): if the token resolves against the active schema it is treated
    # as a column reference; otherwise it is rewritten as a string Literal.
    # Qualified references such as ``t."col"`` are *not* marked because
    # their qualifier disambiguates them as identifiers.
    quoted: bool = False


@dataclass
class Literal:
    value: str | int | float
    data_type: _Lit["string", "integer", "real"]
    pos: Position


@dataclass
class BinaryOp:
    op: str
    left: Expression
    right: Expression
    pos: Position


@dataclass
class UnaryOp:
    op: str
    operand: Expression
    pos: Position


@dataclass
class FunctionCall:
    name: str
    args: list[Expression]
    pos: Position


@dataclass
class Aggregate:
    function: AggregateFunction
    column: ColumnRef | Literal
    distinct: bool
    pos: Position


@dataclass
class InList:
    left: Expression
    values: list[Expression]
    pos: Position


@dataclass
class InSubquery:
    left: Expression
    subquery: SelectStatement
    pos: Position


@dataclass
class ExistsExpr:
    subquery: SelectStatement
    pos: Position


# --- ORDER BY key ----------------------------------------------------------


@dataclass
class OrderKey:
    expr: Expression
    direction: _Lit["asc", "desc"]


# --- Table sources ---------------------------------------------------------


@dataclass
class TableRef:
    name: str
    alias: str | None
    pos: Position


@dataclass
class DerivedTable:
    """A parenthesised subquery used as a FROM source.

    SQL example::

        FROM (SELECT m.id, COUNT(*) AS n FROM x WHERE … GROUP BY m.id) AS t

    The inner :class:`SelectStatement` is parsed independently and
    aliased to ``alias`` in the enclosing scope. The translator
    inlines the subquery's body into the outer DRC, exposing each
    of the subquery's projected columns under ``alias.col`` lookups.

    Unlike :class:`TableRef`, ``alias`` is required — SQL requires
    every derived table to be aliased.
    """

    subquery: "SelectStatement"
    alias: str
    pos: Position


@dataclass
class InnerJoin:
    right: "TableRef | DerivedTable"
    on: Expression
    pos: Position


@dataclass
class JoinChain:
    base: "TableRef | DerivedTable"
    joins: list[InnerJoin]


TableSource = Union[TableRef, DerivedTable, JoinChain]


# --- SELECT statement ------------------------------------------------------


@dataclass
class SelectItem:
    expr: Expression
    alias: str | None


@dataclass
class SelectStatement:
    select_list: list[SelectItem]
    from_source: TableSource
    where: Expression | None
    group_by: list[ColumnRef]
    order_by: list[OrderKey]
    limit: int | None
    pos: Position


# --- Expression union ------------------------------------------------------

Expression = Union[
    ColumnRef,
    Literal,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    Aggregate,
    InList,
    InSubquery,
    ExistsExpr,
]


__all__ = [
    "Position",
    "AggregateFunction",
    "ColumnRef",
    "Literal",
    "BinaryOp",
    "UnaryOp",
    "FunctionCall",
    "Aggregate",
    "InList",
    "InSubquery",
    "ExistsExpr",
    "OrderKey",
    "TableRef",
    "DerivedTable",
    "InnerJoin",
    "JoinChain",
    "TableSource",
    "SelectItem",
    "SelectStatement",
    "Expression",
]
