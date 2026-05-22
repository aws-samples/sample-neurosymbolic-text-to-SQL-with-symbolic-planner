"""SQL converter: transforms an OperationTree into a SQL SELECT statement.

Uses a flattening approach: collects all tables, joins, and WHERE conditions
from the operation tree, then emits a single flat SQL statement rather than
deeply nested subqueries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from text_to_sql_planner.types.operation_tree import (
    OperationTree,
    OperationNode,
    TableLeafNode,
    OperatorNode,
)
from text_to_sql_planner.types.operators import (
    SelectionParams,
    JoinParams,
    ProjectionParams,
    CartesianProductParams,
    UnionParams,
    DivisionParams,
)
from text_to_sql_planner.types.drc import (
    ComparisonNode,
    LogicalConnectiveNode,
    NotNode,
    VariableRefNode,
    LiteralNode,
    MembershipNode,
    FunctionCallNode,
    DRCCondition,
)


@dataclass
class SQLSuccess:
    sql: str


@dataclass
class SQLFailure:
    error: str


SQLResult = Union[SQLSuccess, SQLFailure]

_AGGREGATE_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


def convert_to_sql(tree: OperationTree) -> SQLResult:
    """Convert an OperationTree into a flat SQL SELECT statement.

    Flattens joins and selections into a single SELECT ... FROM ... JOIN ... WHERE ...
    """
    if tree is None or tree.root is None:
        return SQLFailure(error="Invalid operation tree: root is None")

    try:
        sql = _convert_node(tree.root)
        return SQLSuccess(sql=sql)
    except _ConversionError as e:
        return SQLFailure(error=str(e))


class _ConversionError(Exception):
    pass


@dataclass
class _FlatQuery:
    """Intermediate representation for flattening joins/selections."""
    select_columns: list[str] = field(default_factory=list)
    from_tables: list[str] = field(default_factory=list)
    join_clauses: list[str] = field(default_factory=list)
    where_conditions: list[str] = field(default_factory=list)

    def to_sql(self) -> str:
        cols = ", ".join(self.select_columns) if self.select_columns else "*"
        parts = [f"SELECT {cols}"]
        parts.append(f"  FROM {self.from_tables[0]}")
        for join in self.join_clauses:
            parts.append(f"  {join}")
        if self.where_conditions:
            parts.append(f"  WHERE {' AND '.join(self.where_conditions)}")
        return "\n".join(parts)


def _convert_node(node: OperationNode) -> str:
    """Convert an operation node to SQL."""
    if node is None:
        raise _ConversionError("Invalid tree: encountered None node")

    if isinstance(node, TableLeafNode):
        return _convert_table_leaf(node)
    elif isinstance(node, OperatorNode):
        return _convert_operator(node)
    else:
        raise _ConversionError(f"Unknown node type: {type(node).__name__}")


def _convert_table_leaf(node: TableLeafNode) -> str:
    """Table leaf → SELECT cols FROM table"""
    if not node.table_name:
        raise _ConversionError("Table leaf has empty table name")
    if node.columns:
        cols = ", ".join(node.columns)
        return f"SELECT {cols} FROM {node.table_name}"
    return f"SELECT * FROM {node.table_name}"


def _convert_operator(node: OperatorNode) -> str:
    """Convert operator node — tries to flatten joins/selections."""
    params = node.params

    if isinstance(params, ProjectionParams):
        return _convert_projection(node, params)
    elif isinstance(params, UnionParams):
        return _convert_union(node, params)
    elif isinstance(params, DivisionParams):
        return _convert_division(node, params)
    else:
        # For joins, selections, cartesian products: flatten into a single query
        flat = _flatten_to_query(node)
        return flat.to_sql()


def _flatten_to_query(node: OperationNode) -> _FlatQuery:
    """Recursively flatten a tree of joins/selections/cartesian products into a flat query."""
    if isinstance(node, TableLeafNode):
        return _FlatQuery(
            select_columns=list(node.columns) if node.columns else ["*"],
            from_tables=[node.table_name],
        )

    if not isinstance(node, OperatorNode):
        raise _ConversionError(f"Cannot flatten node type: {type(node).__name__}")

    params = node.params
    inputs = node.inputs

    if isinstance(params, SelectionParams):
        # Flatten the input, add the WHERE condition
        if not inputs:
            raise _ConversionError("Selection requires exactly 1 input")
        if params.condition is None:
            raise _ConversionError("Selection requires a condition")
        flat = _flatten_to_query(inputs[0])
        condition_sql = _condition_to_sql(params.condition)
        if condition_sql:
            flat.where_conditions.append(condition_sql)
        # Update select columns to match this node's output
        if node.output_columns:
            flat.select_columns = list(node.output_columns)
        return flat

    elif isinstance(params, JoinParams):
        if len(inputs) < 2:
            raise _ConversionError("Join requires exactly 2 inputs")
        if not params.join_columns:
            raise _ConversionError("Join requires at least one join column")

        left_flat = _flatten_to_query(inputs[0])
        right_flat = _flatten_to_query(inputs[1])

        # The right side becomes a JOIN clause
        right_table = right_flat.from_tables[0] if right_flat.from_tables else "?"
        on_conditions = [f"{left_flat.from_tables[0]}.{col} = {right_table}.{col}"
                         for col in params.join_columns]
        on_clause = " AND ".join(on_conditions)

        # Merge
        result = _FlatQuery(
            select_columns=list(node.output_columns) if node.output_columns else ["*"],
            from_tables=left_flat.from_tables,
            join_clauses=left_flat.join_clauses + right_flat.join_clauses + [
                f"JOIN {right_table} ON {on_clause}"
            ],
            where_conditions=left_flat.where_conditions + right_flat.where_conditions,
        )
        return result

    elif isinstance(params, CartesianProductParams):
        if len(inputs) < 2:
            raise _ConversionError("Cartesian product requires exactly 2 inputs")

        left_flat = _flatten_to_query(inputs[0])
        right_flat = _flatten_to_query(inputs[1])

        right_table = right_flat.from_tables[0] if right_flat.from_tables else "?"

        result = _FlatQuery(
            select_columns=list(node.output_columns) if node.output_columns else ["*"],
            from_tables=left_flat.from_tables,
            join_clauses=left_flat.join_clauses + right_flat.join_clauses + [
                f"CROSS JOIN {right_table}"
            ],
            where_conditions=left_flat.where_conditions + right_flat.where_conditions,
        )
        return result

    else:
        # For other operators that can't be flattened, convert to subquery
        inner_sql = _convert_node(node)
        return _FlatQuery(
            select_columns=list(node.output_columns) if node.output_columns else ["*"],
            from_tables=[f"({inner_sql}) AS sub"],
        )


def _convert_projection(node: OperatorNode, params: ProjectionParams) -> str:
    """Projection → SELECT specific columns FROM (flattened inner query)"""
    inputs = node.inputs
    if not inputs:
        raise _ConversionError("Projection requires exactly 1 input")
    if not params.columns:
        raise _ConversionError("Projection requires at least one column")

    _validate_projection_columns(params.columns, inputs[0])

    # Flatten the inner query
    flat = _flatten_to_query(inputs[0])

    # Replace select columns with the projection columns
    flat.select_columns = [_col_to_sql(col) for col in params.columns]

    return flat.to_sql()


def _convert_union(node: OperatorNode, params: UnionParams) -> str:
    """Union → (...) UNION (...)"""
    inputs = node.inputs
    if len(inputs) < 2:
        raise _ConversionError("Union requires exactly 2 inputs")

    left_sql = _convert_node(inputs[0])
    right_sql = _convert_node(inputs[1])

    return f"({left_sql}) UNION ({right_sql})"


def _convert_division(node: OperatorNode, params: DivisionParams) -> str:
    """Division → double NOT EXISTS pattern"""
    inputs = node.inputs
    if len(inputs) < 2:
        raise _ConversionError("Division requires exactly 2 inputs")

    left_cols = _get_node_columns(inputs[0])
    right_cols = _get_node_columns(inputs[1])

    if not right_cols:
        raise _ConversionError("Division: right input has no columns")

    result_cols = [c for c in left_cols if c not in right_cols]
    if not result_cols:
        raise _ConversionError("Division: no result columns")

    left_table = _get_table_name(inputs[0])
    right_table = _get_table_name(inputs[1])

    result_cols_sql = ", ".join(result_cols)

    inner_conditions = []
    for col in result_cols:
        inner_conditions.append(f"t3.{col} = t1.{col}")
    for col in right_cols:
        inner_conditions.append(f"t3.{col} = t2.{col}")
    inner_where = " AND ".join(inner_conditions)

    return (
        f"SELECT DISTINCT {result_cols_sql} FROM {left_table} t1 "
        f"WHERE NOT EXISTS ("
        f"SELECT * FROM {right_table} t2 "
        f"WHERE NOT EXISTS ("
        f"SELECT * FROM {left_table} t3 "
        f"WHERE {inner_where}"
        f"))"
    )


# --- Helpers ---


def _get_table_name(node: OperationNode) -> str:
    """Get the base table name from a node (for simple cases)."""
    if isinstance(node, TableLeafNode):
        return node.table_name
    # For complex nodes, fall back to subquery
    sql = _convert_node(node)
    return f"({sql})"


def _get_node_columns(node: OperationNode) -> list[str]:
    """Get the output columns of a node."""
    if isinstance(node, TableLeafNode):
        return node.columns
    elif isinstance(node, OperatorNode):
        return node.output_columns
    return []


def _col_to_sql(col: str) -> str:
    """Convert a column spec to SQL syntax (handles aggregates)."""
    col = col.strip()
    if col.startswith("(") and col.endswith(")"):
        inner = col[1:-1].strip()
        parts = inner.split()
        if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
            return f"{parts[0]}({parts[1]})"
    for func in _AGGREGATE_FUNCS:
        if col.startswith(f"{func}(") and col.endswith(")"):
            return col
    parts = col.split()
    if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
        return f"{parts[0]}({parts[1]})"
    return col


def _extract_underlying_column(col: str) -> str:
    """Extract the underlying column name from a column spec."""
    col = col.strip()
    if col.startswith("(") and col.endswith(")"):
        inner = col[1:-1].strip()
        parts = inner.split()
        if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
            return parts[1]
    for func in _AGGREGATE_FUNCS:
        if col.startswith(f"{func}(") and col.endswith(")"):
            return col[len(func) + 1:-1].strip()
    parts = col.split()
    if len(parts) == 2 and parts[0] in _AGGREGATE_FUNCS:
        return parts[1]
    return col


def _validate_projection_columns(columns: list[str], input_node: OperationNode) -> None:
    """Validate that projection columns exist in the input node."""
    available = _get_node_columns(input_node)
    if not available:
        return
    for col in columns:
        underlying = _extract_underlying_column(col)
        if underlying not in available:
            raise _ConversionError(
                f"Projection column '{col}' (underlying: '{underlying}') not found in input columns: {available}"
            )


def _condition_to_sql(condition: DRCCondition) -> str:
    """Convert a DRC condition to a SQL WHERE clause expression."""
    if condition is None:
        raise _ConversionError("Condition is None")

    if isinstance(condition, ComparisonNode):
        left = _condition_to_sql(condition.left)
        right = _condition_to_sql(condition.right)
        return f"{left} {condition.operator} {right}"

    elif isinstance(condition, LogicalConnectiveNode):
        left = _condition_to_sql(condition.left)
        right = _condition_to_sql(condition.right)
        op = condition.operator.upper()
        if op == "IMPLIES":
            return f"(NOT ({left}) OR {right})"
        return f"({left} {op} {right})"

    elif isinstance(condition, NotNode):
        operand = _condition_to_sql(condition.operand)
        return f"NOT ({operand})"

    elif isinstance(condition, VariableRefNode):
        return condition.name

    elif isinstance(condition, LiteralNode):
        if condition.data_type == "string":
            escaped = str(condition.value).replace("'", "''")
            return f"'{escaped}'"
        else:
            return str(condition.value)

    elif isinstance(condition, MembershipNode):
        return ""

    elif isinstance(condition, FunctionCallNode):
        if condition.function == "CURRENT_DATE":
            return "CURRENT_DATE"
        elif condition.function == "DATE_SUB" and len(condition.arguments) == 2:
            base = _condition_to_sql(condition.arguments[0])
            days = _condition_to_sql(condition.arguments[1])
            return f"{base} - INTERVAL '{days} days'"
        elif condition.function == "DATE_ADD" and len(condition.arguments) == 2:
            base = _condition_to_sql(condition.arguments[0])
            days = _condition_to_sql(condition.arguments[1])
            return f"{base} + INTERVAL '{days} days'"
        elif condition.function == "DATEDIFF" and len(condition.arguments) == 2:
            left = _condition_to_sql(condition.arguments[0])
            right = _condition_to_sql(condition.arguments[1])
            return f"({left} - {right})"
        else:
            if not condition.arguments:
                return condition.function
            args = ", ".join(_condition_to_sql(arg) for arg in condition.arguments)
            return f"{condition.function}({args})"

    else:
        raise _ConversionError(f"Unsupported condition type: {type(condition).__name__}")
