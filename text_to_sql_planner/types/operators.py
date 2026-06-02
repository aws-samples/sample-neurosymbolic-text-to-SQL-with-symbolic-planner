"""Relational Algebra operator types and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Literal

from text_to_sql_planner.types.drc import DRCExpression, DRCCondition


RAOperatorType = Literal[
    "selection", "join", "projection",
    "cartesian_product", "union", "difference", "division",
    "rename",
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
