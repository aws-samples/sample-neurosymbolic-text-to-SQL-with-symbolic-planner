"""Relational Algebra operator types and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Literal

from text_to_sql_planner.types.drc import DRCExpression, DRCCondition


RAOperatorType = Literal[
    "selection", "join", "projection",
    "cartesian_product", "union", "difference", "division",
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
