"""Operation tree types representing proven-correct RA operation sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, Literal

from text_to_sql_planner.types.drc import DRCExpression
from text_to_sql_planner.types.operators import RAOperatorType, OperatorParams


@dataclass
class OperationTree:
    root: OperationNode = None  # type: ignore


@dataclass
class TableLeafNode:
    """Leaf: a base table relation"""
    type: Literal["table_leaf"] = "table_leaf"
    table_name: str = ""
    columns: list[str] = field(default_factory=list)
    expression: DRCExpression = None  # type: ignore


@dataclass
class OperatorNode:
    """Internal node: an RA operator application"""
    type: Literal["operator_node"] = "operator_node"
    operator: RAOperatorType = "selection"
    params: OperatorParams = None  # type: ignore
    inputs: list[OperationNode] = field(default_factory=list)
    output_expression: DRCExpression = None  # type: ignore
    output_columns: list[str] = field(default_factory=list)


OperationNode = Union[TableLeafNode, OperatorNode]
