"""DRC (Domain Relational Calculus) AST type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Literal


# Top-level DRC expression: {result_vars | condition}
@dataclass
class DRCExpression:
    result_variables: list[ResultVariable] = field(default_factory=list)
    condition: DRCCondition = None  # type: ignore


# Result variable can be a plain column or an aggregate
@dataclass
class ColumnVariable:
    type: Literal["column"] = "column"
    name: str = ""


@dataclass
class AggregateVariable:
    type: Literal["aggregate"] = "aggregate"
    function: AggregateFunction = "COUNT"
    column: str = ""


ResultVariable = Union[ColumnVariable, AggregateVariable]

AggregateFunction = Literal["COUNT", "SUM", "AVG", "MIN", "MAX"]


# Condition nodes (recursive)

@dataclass
class QuantifierNode:
    """Quantifiers: ∀ and ∃"""
    type: Literal["quantifier"] = "quantifier"
    kind: Literal["forall", "exists"] = "forall"
    variables: list[str] = field(default_factory=list)
    relation: str = ""  # relation name for ∈
    body: DRCCondition = None  # type: ignore


@dataclass
class LogicalConnectiveNode:
    """Binary logical connectives: ∧, ∨, →"""
    type: Literal["logical_connective"] = "logical_connective"
    operator: Literal["and", "or", "implies"] = "and"
    left: DRCCondition = None  # type: ignore
    right: DRCCondition = None  # type: ignore


@dataclass
class NotNode:
    """Negation: ¬"""
    type: Literal["not"] = "not"
    operand: DRCCondition = None  # type: ignore


@dataclass
class ComparisonNode:
    """Comparison: =, !=, <, >, <=, >="""
    type: Literal["comparison"] = "comparison"
    operator: Literal["=", "!=", "<", ">", "<=", ">="] = "="
    left: DRCCondition = None  # type: ignore
    right: DRCCondition = None  # type: ignore


@dataclass
class MembershipNode:
    """Relation membership: (in (vars...) RelationName)"""
    type: Literal["membership"] = "membership"
    variables: list[str] = field(default_factory=list)
    relation: str = ""


@dataclass
class ArithmeticNode:
    """Arithmetic: +, -, *, /"""
    type: Literal["arithmetic"] = "arithmetic"
    operator: Literal["+", "-", "*", "/"] = "+"
    left: DRCCondition = None  # type: ignore
    right: DRCCondition = None  # type: ignore


@dataclass
class LiteralNode:
    """String or numeric literal"""
    type: Literal["literal"] = "literal"
    value: Union[str, int, float] = ""
    data_type: Literal["string", "number"] = "string"


@dataclass
class VariableRefNode:
    """Variable reference"""
    type: Literal["variable_ref"] = "variable_ref"
    name: str = ""


DRCCondition = Union[
    QuantifierNode,
    LogicalConnectiveNode,
    NotNode,
    ComparisonNode,
    MembershipNode,
    ArithmeticNode,
    LiteralNode,
    VariableRefNode,
]

DRCNode = Union[DRCExpression, DRCCondition]
