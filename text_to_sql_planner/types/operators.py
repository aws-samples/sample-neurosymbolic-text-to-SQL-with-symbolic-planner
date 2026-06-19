"""Relational Algebra operator types and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Literal

from text_to_sql_planner.types.drc import DRCExpression, DRCCondition


RAOperatorType = Literal[
    "selection", "join", "projection",
    "cartesian_product", "union", "difference", "division",
    "rename",
    "aggregate",
    "ratio",
    "anti_join",
]


@dataclass
class SelectionParams:
    type: Literal["selection"] = "selection"
    condition: DRCCondition = None  # type: ignore


@dataclass
class JoinParams:
    type: Literal["join"] = "join"
    join_columns: list[str] = field(default_factory=list)


@dataclass
class ProjectionParams:
    type: Literal["projection"] = "projection"
    columns: list[str] = field(default_factory=list)


@dataclass
class CartesianProductParams:
    type: Literal["cartesian_product"] = "cartesian_product"


@dataclass
class UnionParams:
    type: Literal["union"] = "union"


@dataclass
class DifferenceParams:
    type: Literal["difference"] = "difference"


@dataclass
class DivisionParams:
    type: Literal["division"] = "division"


@dataclass
class RenameParams:
    """Rename one or more columns of a relation.

    ``mapping`` is a dictionary of ``old_name -> new_name`` entries.
    Each ``old_name`` must exist in the input relation's columns;
    each ``new_name`` must NOT collide with any other input column
    that isn't being renamed away in the same operation. The
    operator preserves the input's row set unchanged — it's a pure
    schema rewrite.

    Why an explicit operator: relational algebra historically uses
    rename (``ρ``) to disambiguate self-joins (e.g.
    ``Performance_Reviews ⋈ ρ_{review_id ← review_id_2}(Performance_Reviews)``).
    Without it, natural join collapses on every shared column and
    the planner has to fall back on cartesian product + selection.
    The planner can use this operator to do the standard "make a
    second copy of T with renamed columns" pattern.
    """

    type: Literal["rename"] = "rename"
    mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class AggregateParams:
    """Aggregate a single-column relation into one aggregate result variable.

    Promotes a :class:`ColumnVariable` result variable to an
    :class:`AggregateVariable` by wrapping it in one of the SQL
    aggregate functions (``COUNT`` / ``SUM`` / ``AVG`` / ``MIN`` /
    ``MAX``). The input must already have *exactly one* result
    variable, of column type, and that variable must be the one
    being aggregated — the operator's job is the structural
    promotion of ``{c | φ}`` into ``{(F c) | φ}``, not a grouped
    aggregation. Group-by aggregation is expressed as
    ``projection [keys, (F col)]`` on the join, exactly as it is
    today; this operator handles the simpler "aggregate the whole
    relation to a single value" case the planner currently has no
    way to express.

    Why this is its own operator: ``rename`` can only relabel a
    column-typed result variable as another column name; it cannot
    change the result variable's *kind*. The planner builds the
    correct underlying relation (e.g. "BountyAmount projected from
    the right join") but then has nowhere to go because every
    further ``rename`` attempt produces a structurally-identical
    DRC and gets de-duplicated. Adding this operator gives the
    planner an explicit way to perform the
    ``ColumnVariable → AggregateVariable`` transition that the
    target DRC requires.
    """

    type: Literal["aggregate"] = "aggregate"
    function: Literal["COUNT", "SUM", "AVG", "MIN", "MAX"] = "COUNT"
    column: str = ""


@dataclass
class RatioParams:
    """Ratio of two aggregates: (/ (F1 col1) (F2 col2)).

    Takes an input with exactly 2 ColumnVariable result variables and
    produces a single ArithmeticResultVariable wrapping two
    AggregateVariables.

    When ``numerator_condition`` is set (a DRC condition lisp string),
    the numerator becomes a CountIfVariable instead of a plain
    AggregateVariable, producing ``(/ (COUNT_IF cond col1) (F2 col2))``.
    """

    type: Literal["ratio"] = "ratio"
    operator: str = "/"  # arithmetic operator: +, -, *, /
    numerator_function: Literal["COUNT", "SUM", "AVG", "MIN", "MAX"] = "COUNT"
    numerator_column: str = ""
    denominator_function: Literal["COUNT", "SUM", "AVG", "MIN", "MAX"] = "COUNT"
    denominator_column: str = ""
    numerator_condition: str | None = None  # optional lisp condition for COUNT_IF
    denominator_condition: str | None = None  # optional lisp condition for COUNT_IF on denominator
    scalar_multiplier: int | float | None = None  # optional scalar (e.g. 100 for percentage)


@dataclass
class AntiJoinParams:
    """Anti-join: rows of the left input whose key is NOT present in the right.

    Always paired with a list of join columns (the keys against which
    presence-in-right is tested). The output column shape equals the
    left input's column shape — anti-join is a *filter* operator, not
    a column-merging join.

    The planner doesn't propose this operator directly (it's not in the
    LLM's selectable RA toolbox). It's introduced post-planning by the
    operation-tree simplifier when it recognises the pattern
    ``Join(T1, Difference(T1, T2), key=k) → AntiJoin(T1, T2, key=k)``.
    """

    type: Literal["anti_join"] = "anti_join"
    join_columns: list[str] = field(default_factory=list)


OperatorParams = Union[
    SelectionParams, JoinParams, ProjectionParams,
    CartesianProductParams, UnionParams, DifferenceParams, DivisionParams,
    RenameParams,
    AggregateParams,
    RatioParams,
    AntiJoinParams,
]


@dataclass
class OperatorApplication:
    operator: RAOperatorType
    inputs: list[DRCExpression] = field(default_factory=list)
    params: OperatorParams = None  # type: ignore


@dataclass
class OperatorSuccess:
    output: DRCExpression = None  # type: ignore


@dataclass
class OperatorFailure:
    error: str = ""


OperatorResult = Union[OperatorSuccess, OperatorFailure]
