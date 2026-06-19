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


@dataclass
class ArithmeticAggregateVariable:
    """Arithmetic expression over two aggregates.

    Represents SQL patterns like ``COUNT(x) / COUNT(DISTINCT y)`` or
    ``SUM(a) * 100.0 / SUM(b)`` that appear as a single result column.

    In the DRC S-expression syntax this is written as::

        (/ (COUNT x) (COUNT y))

    in the result-variable position. The SQL converter emits
    ``AGG1(col1) <op> AGG2(col2)`` in the SELECT list.

    The equivalence checker compares these structurally — two
    ArithmeticAggregateVariable values are equivalent iff they have
    the same operator and the same left/right aggregates (by function
    name and column name).
    """
    type: Literal["arithmetic_aggregate"] = "arithmetic_aggregate"
    operator: Literal["+", "-", "*", "/"] = "/"
    left: "ResultVariable" = field(default_factory=lambda: AggregateVariable())
    right: "ResultVariable" = field(default_factory=lambda: AggregateVariable())


@dataclass
class ConditionalAggregateVariable:
    """Conditional aggregation: COUNT/SUM only rows matching a condition.

    Represents SQL patterns like::

        COUNT(CASE WHEN Diagnosis LIKE '%SLE%' THEN ID END)
        SUM(CASE WHEN status = 'A' THEN amount ELSE 0 END)

    In the DRC S-expression syntax::

        (COUNT_IF (LIKE Diagnosis "%SLE%") ID)

    The condition is a DRC condition node (same type system as the
    body of a ``where`` clause). The column is counted/summed only
    for rows satisfying the condition.

    The SQL converter emits::

        COUNT(CASE WHEN <condition_sql> THEN <col> END)
    """
    type: Literal["conditional_aggregate"] = "conditional_aggregate"
    function: AggregateFunction = "COUNT"
    column: str = ""
    condition: "DRCCondition" = None  # type: ignore


@dataclass
class ScalarLiteralVariable:
    """A numeric literal in result-variable position (e.g. 100 in ``(* (/ ...) 100)``)."""
    type: Literal["scalar_literal"] = "scalar_literal"
    value: Union[int, float] = 0


ResultVariable = Union[ColumnVariable, AggregateVariable, ArithmeticAggregateVariable, ConditionalAggregateVariable, ScalarLiteralVariable]

AggregateFunction = Literal["COUNT", "SUM", "AVG", "MIN", "MAX"]


# Condition nodes (recursive)

@dataclass
class QuantifierNode:
    """Quantifiers: ∀ and ∃ — bind variables, body is the condition."""
    type: Literal["quantifier"] = "quantifier"
    kind: Literal["forall", "exists"] = "forall"
    variables: list[str] = field(default_factory=list)
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
class IsNotNullNode:
    """IS NOT NULL predicate over a single column reference.

    Represents the s-expression ``(is-not-null col)`` and the SQL form
    ``col IS NOT NULL``. This is a first-class DRC AST node consistent
    with the existing ``rewrite_null_checks`` preprocessing pass: the
    SMT translator emits the same SMT for ``IsNotNullNode(column=col)``
    as it does for the rewritten form of
    ``FunctionCallNode("IS_NOT_NULL", [VariableRefNode(col)])``.
    """
    type: Literal["is_not_null"] = "is_not_null"
    column: str = ""


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


@dataclass
class FunctionCallNode:
    """Built-in function call: CURRENT_DATE, DATE_SUB, YEAR, etc."""
    type: Literal["function_call"] = "function_call"
    function: str = ""
    arguments: list[DRCCondition] = field(default_factory=list)


DRCCondition = Union[
    QuantifierNode,
    LogicalConnectiveNode,
    NotNode,
    ComparisonNode,
    MembershipNode,
    ArithmeticNode,
    LiteralNode,
    VariableRefNode,
    FunctionCallNode,
    IsNotNullNode,
]

DRCNode = Union[DRCExpression, DRCCondition]


# ---------------------------------------------------------------------------
# Extended DRC: order/limit wrappers (non-relational layer)
# ---------------------------------------------------------------------------
#
# Core DRC is set-based, so it has no notion of row order or result size.
# To support questions like "the top 5 most-compensated employees" we wrap
# a core DRC expression in non-DRC operators. The wrappers compose and the
# pipeline always normalizes a query as
#
#     QueryExpression = LIMIT? · ORDER_BY? · DRCExpression
#
# i.e. an optional LIMIT outside an optional ORDER_BY outside the
# set-comprehension. Either wrapper can be omitted.
#
# The wrappers are deliberately *not* part of ``DRCCondition`` — they live
# at the query level because they cannot be expressed in first-order
# predicate logic over relations. Equivalence checking peels them off and
# compares the inner DRC; the SQL converter emits them as ``ORDER BY`` /
# ``LIMIT`` clauses on the outermost SELECT.


SortDirection = Literal["asc", "desc"]


@dataclass
class SortCriterion:
    """A single ORDER BY key.

    ``column`` must reference a result variable of the inner DRC (a plain
    column name, or the underlying column of an aggregate result variable
    such as ``column='salary'`` for ``(SUM salary)``).

    ``aggregate`` is set when the key is an aggregate function over
    ``column`` (mirroring ``AggregateVariable``); otherwise it's ``None``
    and the key is the bare column.
    """

    column: str = ""
    direction: SortDirection = "asc"
    aggregate: Union[AggregateFunction, None] = None


@dataclass
class OrderByExpression:
    """ORDER BY wrapper around a DRC set.

    ``criteria`` is a non-empty list of sort keys, applied in order
    (first key is primary, second tie-breaks, etc.).
    """

    criteria: list[SortCriterion] = field(default_factory=list)
    inner: DRCExpression = None  # type: ignore


@dataclass
class LimitExpression:
    """LIMIT wrapper around an ordered (or set) DRC expression.

    ``n`` is the (positive) row count cap. ``inner`` is typically an
    ``OrderByExpression`` — applying ``LIMIT`` to an unordered set is
    non-deterministic but syntactically allowed for completeness.
    """

    n: int = 0
    inner: Union[OrderByExpression, DRCExpression] = None  # type: ignore


QueryExpression = Union[LimitExpression, OrderByExpression, DRCExpression]


def query_inner_drc(query: QueryExpression) -> DRCExpression:
    """Strip any LIMIT/ORDER BY wrappers and return the core DRCExpression.

    Raises ``TypeError`` if the eventual core is not a ``DRCExpression``.
    """
    while not isinstance(query, DRCExpression):
        if isinstance(query, LimitExpression) or isinstance(query, OrderByExpression):
            query = query.inner
        else:
            raise TypeError(
                f"Unexpected query node type: {type(query).__name__}"
            )
    return query


def query_order_by(query: QueryExpression) -> Union[OrderByExpression, None]:
    """Return the ORDER BY layer of ``query`` if present, else ``None``."""
    if isinstance(query, LimitExpression):
        return query.inner if isinstance(query.inner, OrderByExpression) else None
    if isinstance(query, OrderByExpression):
        return query
    return None


def query_limit(query: QueryExpression) -> Union[LimitExpression, None]:
    """Return the LIMIT layer of ``query`` if present, else ``None``."""
    return query if isinstance(query, LimitExpression) else None

# Backward-compatibility alias used by operators/ratio.py
ArithmeticResultVariable = ArithmeticAggregateVariable
CountIfVariable = ConditionalAggregateVariable
