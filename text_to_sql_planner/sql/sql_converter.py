"""SQL converter: transforms an OperationTree into a SQL SELECT statement."""

from __future__ import annotations

from dataclasses import dataclass
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
    DRCCondition,
)


@dataclass
class SQLSuccess:
    sql: str


@dataclass
class SQLFailure:
    error: str


SQLResult = Union[SQLSuccess, SQLFailure]


def convert_to_sql(tree: OperationTree) -> SQLResult:
    """Convert an OperationTree into a SQL SELECT statement.

    Performs bottom-up traversal of the operation tree, mapping each
    RA operator to its SQL equivalent.

    Args:
        tree: The operation tree to convert.

    Returns:
        SQLSuccess with the SQL string, or SQLFailure with an error message.
    """
    if tree is None or tree.root is None:
        return SQLFailure(error="Invalid operation tree: root is None")

    try:
        sql = _convert_node(tree.root)
        return SQLSuccess(sql=sql)
    except _ConversionError as e:
        return SQLFailure(error=str(e))


class _ConversionError(Exception):
    """Internal exception for conversion errors."""
    pass


def _convert_node(node: OperationNode) -> str:
    """Recursively convert an operation node to SQL (bottom-up)."""
    if node is None:
        raise _ConversionError("Invalid tree: encountered None node")

    if isinstance(node, TableLeafNode):
        return _convert_table_leaf(node)
    elif isinstance(node, OperatorNode):
        return _convert_operator(node)
    else:
        raise _ConversionError(f"Unknown node type: {type(node).__name__}")


def _convert_table_leaf(node: TableLeafNode) -> str:
    """Convert a table leaf node to a SQL SELECT statement."""
    if not node.table_name:
        raise _ConversionError("Table leaf has empty table name")
    if node.columns:
        cols = ", ".join(node.columns)
        return f"SELECT {cols} FROM {node.table_name}"
    return f"SELECT * FROM {node.table_name}"


def _convert_operator(node: OperatorNode) -> str:
    """Convert an operator node to SQL based on operator type."""
    params = node.params
    inputs = node.inputs

    if isinstance(params, SelectionParams):
        return _convert_selection(node, params, inputs)
    elif isinstance(params, ProjectionParams):
        return _convert_projection(node, params, inputs)
    elif isinstance(params, JoinParams):
        return _convert_join(node, params, inputs)
    elif isinstance(params, CartesianProductParams):
        return _convert_cartesian_product(node, params, inputs)
    elif isinstance(params, UnionParams):
        return _convert_union(node, params, inputs)
    elif isinstance(params, DivisionParams):
        return _convert_division(node, params, inputs)
    else:
        raise _ConversionError(f"Unknown operator params type: {type(params).__name__}")


def _convert_selection(
    node: OperatorNode, params: SelectionParams, inputs: list[OperationNode]
) -> str:
    """Selection (σ) → SELECT * FROM (...) WHERE condition"""
    if not inputs:
        raise _ConversionError("Selection requires exactly 1 input")
    if params.condition is None:
        raise _ConversionError("Selection requires a condition")

    input_sql = _convert_node(inputs[0])
    condition_sql = _condition_to_sql(params.condition)

    source = _wrap_as_source(input_sql, "t")
    return f"SELECT * FROM {source} WHERE {condition_sql}"


def _convert_projection(
    node: OperatorNode, params: ProjectionParams, inputs: list[OperationNode]
) -> str:
    """Projection (π) → SELECT col1, col2 FROM (...)"""
    if not inputs:
        raise _ConversionError("Projection requires exactly 1 input")
    if not params.columns:
        raise _ConversionError("Projection requires at least one column")

    # Validate columns against output_columns of input
    _validate_projection_columns(params.columns, inputs[0])

    input_sql = _convert_node(inputs[0])
    columns_sql = ", ".join(params.columns)

    source = _wrap_as_source(input_sql, "t")
    return f"SELECT {columns_sql} FROM {source}"


def _convert_join(
    node: OperatorNode, params: JoinParams, inputs: list[OperationNode]
) -> str:
    """Join (⋈) → SELECT ... FROM (...) JOIN (...) ON condition"""
    if len(inputs) < 2:
        raise _ConversionError("Join requires exactly 2 inputs")
    if not params.join_columns:
        raise _ConversionError("Join requires at least one join column")

    left_sql = _convert_node(inputs[0])
    right_sql = _convert_node(inputs[1])

    left_source = _wrap_as_source(left_sql, "t1")
    right_source = _wrap_as_source(right_sql, "t2")

    # Build ON clause
    on_conditions = []
    for col in params.join_columns:
        on_conditions.append(f"t1.{col} = t2.{col}")
    on_clause = " AND ".join(on_conditions)

    # Build column list: all columns from output_columns or use *
    if node.output_columns:
        columns_sql = ", ".join(node.output_columns)
    else:
        columns_sql = "*"

    return f"SELECT {columns_sql} FROM {left_source} JOIN {right_source} ON {on_clause}"


def _convert_cartesian_product(
    node: OperatorNode, params: CartesianProductParams, inputs: list[OperationNode]
) -> str:
    """Cartesian Product (×) → SELECT ... FROM (...) CROSS JOIN (...)"""
    if len(inputs) < 2:
        raise _ConversionError("Cartesian product requires exactly 2 inputs")

    left_sql = _convert_node(inputs[0])
    right_sql = _convert_node(inputs[1])

    left_source = _wrap_as_source(left_sql, "t1")
    right_source = _wrap_as_source(right_sql, "t2")

    # Build column list from output_columns or use *
    if node.output_columns:
        columns_sql = ", ".join(node.output_columns)
    else:
        columns_sql = "*"

    return f"SELECT {columns_sql} FROM {left_source} CROSS JOIN {right_source}"


def _convert_union(
    node: OperatorNode, params: UnionParams, inputs: list[OperationNode]
) -> str:
    """Union (∪) → (...) UNION (...)"""
    if len(inputs) < 2:
        raise _ConversionError("Union requires exactly 2 inputs")

    left_sql = _convert_node(inputs[0])
    right_sql = _convert_node(inputs[1])

    # Wrap each side as a full SELECT if it's just a table name
    left_select = _ensure_select(left_sql, inputs[0])
    right_select = _ensure_select(right_sql, inputs[1])

    return f"({left_select}) UNION ({right_select})"


def _convert_division(
    node: OperatorNode, params: DivisionParams, inputs: list[OperationNode]
) -> str:
    """Division (÷) → SELECT ... FROM ... WHERE NOT EXISTS (SELECT ... FROM ... WHERE NOT EXISTS (...))"""
    if len(inputs) < 2:
        raise _ConversionError("Division requires exactly 2 inputs")

    left_sql = _convert_node(inputs[0])
    right_sql = _convert_node(inputs[1])

    # Determine columns: result columns are those in left but not in right
    left_cols = _get_node_columns(inputs[0])
    right_cols = _get_node_columns(inputs[1])

    if not right_cols:
        raise _ConversionError("Division: right input has no columns")

    result_cols = [c for c in left_cols if c not in right_cols]
    if not result_cols:
        raise _ConversionError("Division: no result columns (all columns are in divisor)")

    result_cols_sql = ", ".join(result_cols)
    right_cols_sql = ", ".join(right_cols)

    left_source = _wrap_as_source(left_sql, "dividend")
    right_source = _wrap_as_source(right_sql, "divisor")

    # Build the double NOT EXISTS pattern
    # SELECT result_cols FROM dividend AS t1
    # WHERE NOT EXISTS (
    #   SELECT * FROM divisor AS t2
    #   WHERE NOT EXISTS (
    #     SELECT * FROM dividend AS t3
    #     WHERE t3.result_col = t1.result_col AND t3.right_col = t2.right_col
    #   )
    # )
    inner_conditions = []
    for col in result_cols:
        inner_conditions.append(f"t3.{col} = t1.{col}")
    for col in right_cols:
        inner_conditions.append(f"t3.{col} = divisor.{col}")
    inner_where = " AND ".join(inner_conditions)

    inner_left_source = _wrap_as_source(left_sql, "t3")

    return (
        f"SELECT DISTINCT {result_cols_sql} FROM {left_source} AS t1 "
        f"WHERE NOT EXISTS ("
        f"SELECT * FROM {right_source} "
        f"WHERE NOT EXISTS ("
        f"SELECT * FROM {inner_left_source} "
        f"WHERE {inner_where}"
        f"))"
    )


# --- Helpers ---


def _wrap_as_source(sql: str, alias: str) -> str:
    """Wrap SQL as a subquery source if it's not a simple table reference.

    If the SQL is a simple 'SELECT cols FROM tablename' (no WHERE, JOIN, etc.),
    just use the table name directly.
    """
    if _is_simple_table_name(sql):
        return sql
    # Check if it's a simple SELECT from a single table (no subquery needed)
    table = _extract_simple_table(sql)
    if table:
        return table
    return f"({sql}) AS {alias}"


def _extract_simple_table(sql: str) -> str | None:
    """If sql is 'SELECT ... FROM tablename' with no WHERE/JOIN/etc, return tablename."""
    import re
    match = re.match(
        r"^SELECT\s+.+?\s+FROM\s+(\w+)$",
        sql.strip(),
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return None


def _is_simple_table_name(sql: str) -> bool:
    """Check if the SQL is just a simple table name (no spaces, keywords, etc.)."""
    return sql.isidentifier()


def _ensure_select(sql: str, node: OperationNode) -> str:
    """Ensure the SQL is a full SELECT statement."""
    if _is_simple_table_name(sql):
        cols = _get_node_columns(node)
        if cols:
            return f"SELECT {', '.join(cols)} FROM {sql}"
        return f"SELECT * FROM {sql}"
    return sql


def _get_node_columns(node: OperationNode) -> list[str]:
    """Get the output columns of a node."""
    if isinstance(node, TableLeafNode):
        return node.columns
    elif isinstance(node, OperatorNode):
        return node.output_columns
    return []


def _validate_projection_columns(columns: list[str], input_node: OperationNode) -> None:
    """Validate that projection columns exist in the input node."""
    available = _get_node_columns(input_node)
    if not available:
        # Can't validate if we don't know the input columns
        return
    for col in columns:
        if col not in available:
            raise _ConversionError(
                f"Projection column '{col}' not found in input columns: {available}"
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
            # p → q is equivalent to NOT p OR q
            return f"(NOT ({left}) OR {right})"
        return f"({left} {op} {right})"

    elif isinstance(condition, NotNode):
        operand = _condition_to_sql(condition.operand)
        return f"NOT ({operand})"

    elif isinstance(condition, VariableRefNode):
        return condition.name

    elif isinstance(condition, LiteralNode):
        if condition.data_type == "string":
            # Escape single quotes in string literals
            escaped = str(condition.value).replace("'", "''")
            return f"'{escaped}'"
        else:
            return str(condition.value)

    elif isinstance(condition, MembershipNode):
        # MembershipNode is structural, not a filter - skip
        return ""

    else:
        raise _ConversionError(f"Unsupported condition type: {type(condition).__name__}")
