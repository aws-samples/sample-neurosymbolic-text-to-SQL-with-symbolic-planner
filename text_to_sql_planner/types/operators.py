"""Relational Algebra operator types and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Literal

from text_to_sql_planner.types.drc import DRCExpression, DRCCondition


RAOperatorType = Literal[
    "selection", "join", "projection",
    "cartesian_product", "union", "difference", "division"
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


OperatorParams = Union[
    SelectionParams, JoinParams, ProjectionParams,
    CartesianProductParams, UnionParams, DifferenceParams, DivisionParams
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
