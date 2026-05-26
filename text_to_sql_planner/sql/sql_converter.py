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
    QuantifierNode,
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


def convert_to_sql(tree: OperationTree, result_variables: list | None = None) -> SQLResult:
    """Convert an OperationTree into a flat SQL SELECT statement.

    If result_variables contains a mix of plain columns and aggregates,
    emits GROUP BY for the plain columns.

    Args:
        tree: The operation tree.
        result_variables: The target DRC expression's result variables (optional).
    """
    if tree is None or tree.root is None:
        return SQLFailure(error="Invalid operation tree: root is None")

    try:
        # If we have result_variables with aggregates, use the flattened approach
        # and add GROUP BY directly
        if result_variables:
            from text_to_sql_planner.types.drc import ColumnVariable, AggregateVariable
            has_aggregates = any(isinstance(rv, AggregateVariable) for rv in result_variables)

            if has_aggregates:
                plain_cols = [rv.name for rv in result_variables if isinstance(rv, ColumnVariable)]
                
                # Build the SELECT column list: plain cols first, then aggregates
                # Flatten the tree to get aliases
                flat = _flatten_to_query(tree.root)

                all_select = []
                for rv in result_variables:
                    if isinstance(rv, ColumnVariable):
                        qualified = _qualify_column_from_aliases(rv.name, flat.table_aliases)
                        all_select.append(qualified)
                    elif isinstance(rv, AggregateVariable):
                        qualified_col = _qualify_column_from_aliases(rv.column, flat.table_aliases)
                        all_select.append(f"{rv.function}({qualified_col})")

                flat.select_columns = all_select
                if plain_cols:
                    # GROUP BY uses the same qualified names as the SELECT columns
                    # Extract the qualified plain columns from all_select (non-aggregate entries)
                    qualified_plain = [s for s in all_select if not any(s.startswith(f"{f}(") for f in _AGGREGATE_FUNCS)]
                    flat.group_by_columns = qualified_plain
                return SQLSuccess(sql=flat.to_sql())

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
    from_tables: list[str] = field(default_factory=list)  # "TableName alias"
    join_clauses: list[str] = field(default_factory=list)
    where_conditions: list[str] = field(default_factory=list)
    group_by_columns: list[str] = field(default_factory=list)
    # Maps table_name -> alias for explicit scoping
    table_aliases: dict[str, str] = field(default_factory=dict)

    def to_sql(self) -> str:
        cols = ", ".join(self.select_columns) if self.select_columns else "*"
        parts = [f"SELECT {cols}"]
        parts.append(f"  FROM {self.from_tables[0]}")
        for join in self.join_clauses:
            parts.append(f"  {join}")
        if self.where_conditions:
            parts.append(f"  WHERE {' AND '.join(self.where_conditions)}")
        if self.group_by_columns:
            parts.append(f"  GROUP BY {', '.join(self.group_by_columns)}")
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
    """Table leaf → SELECT alias.cols FROM table alias"""
    if not node.table_name:
        raise _ConversionError("Table leaf has empty table name")
    alias = _make_alias(node.table_name)
    if node.columns:
        cols = ", ".join(f"{alias}.{col}" for col in node.columns)
        return f"SELECT {cols} FROM {node.table_name} {alias}"
    return f"SELECT * FROM {node.table_name} {alias}"


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
        alias = _make_alias(node.table_name)
        qualified_cols = [f"{alias}.{col}" for col in node.columns] if node.columns else ["*"]
        return _FlatQuery(
            select_columns=qualified_cols,
            from_tables=[f"{node.table_name} {alias}"],
            table_aliases={node.table_name: alias},
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
        # Convert condition with alias context
        condition_sql = _condition_to_sql_with_aliases(params.condition, flat.table_aliases)
        if condition_sql:
            flat.where_conditions.append(condition_sql)
        # Update select columns to match this node's output
        if node.output_columns:
            flat.select_columns = [_qualify_column(col, flat.table_aliases, inputs) for col in node.output_columns]
        return flat

    elif isinstance(params, JoinParams):
        if len(inputs) < 2:
            raise _ConversionError("Join requires exactly 2 inputs")
        if not params.join_columns:
            raise _ConversionError("Join requires at least one join column")

        left_flat = _flatten_to_query(inputs[0])
        right_flat = _flatten_to_query(inputs[1])

        # Resolve alias conflicts between left and right
        left_aliases_set = set(left_flat.table_aliases.values())
        right_alias = list(right_flat.table_aliases.values())[0] if right_flat.table_aliases else "t2"

        if right_alias in left_aliases_set:
            # Conflict — generate a new unique alias for the right side
            right_table_name = list(right_flat.table_aliases.keys())[0] if right_flat.table_aliases else "t"
            new_alias = _make_alias(right_table_name, left_aliases_set)
            # Update right_flat's from_tables and table_aliases
            old_alias = right_alias
            right_alias = new_alias
            right_flat.from_tables = [f.replace(f" {old_alias}", f" {new_alias}") for f in right_flat.from_tables]
            right_flat.table_aliases = {k: (new_alias if v == old_alias else v) for k, v in right_flat.table_aliases.items()}

        # Get aliases for ON clause
        left_alias = list(left_flat.table_aliases.values())[0] if left_flat.table_aliases else "t1"
        right_from = right_flat.from_tables[0] if right_flat.from_tables else "?"

        on_conditions = [f"{left_alias}.{col} = {right_alias}.{col}"
                         for col in params.join_columns]
        on_clause = " AND ".join(on_conditions)

        # Qualify output columns
        output_cols = []
        if node.output_columns:
            # Try to qualify each output column with the correct alias
            all_aliases = {**left_flat.table_aliases, **right_flat.table_aliases}
            for col in node.output_columns:
                qualified = _qualify_column(col, all_aliases, inputs)
                output_cols.append(qualified)
        else:
            output_cols = ["*"]

        result = _FlatQuery(
            select_columns=output_cols,
            from_tables=left_flat.from_tables,
            join_clauses=left_flat.join_clauses + right_flat.join_clauses + [
                f"JOIN {right_from} ON {on_clause}"
            ],
            where_conditions=left_flat.where_conditions + right_flat.where_conditions,
            table_aliases={**left_flat.table_aliases, **right_flat.table_aliases},
        )
        return result

    elif isinstance(params, CartesianProductParams):
        if len(inputs) < 2:
            raise _ConversionError("Cartesian product requires exactly 2 inputs")

        left_flat = _flatten_to_query(inputs[0])
        right_flat = _flatten_to_query(inputs[1])

        right_from = right_flat.from_tables[0] if right_flat.from_tables else "?"

        # Qualify output columns
        output_cols = []
        if node.output_columns:
            all_aliases = {**left_flat.table_aliases, **right_flat.table_aliases}
            for col in node.output_columns:
                qualified = _qualify_column(col, all_aliases, inputs)
                output_cols.append(qualified)
        else:
            output_cols = ["*"]

        result = _FlatQuery(
            select_columns=output_cols,
            from_tables=left_flat.from_tables,
            join_clauses=left_flat.join_clauses + right_flat.join_clauses + [
                f"CROSS JOIN {right_from}"
            ],
            where_conditions=left_flat.where_conditions + right_flat.where_conditions,
            table_aliases={**left_flat.table_aliases, **right_flat.table_aliases},
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

    left_indented = _indent_sql(left_sql, indent=2)
    right_indented = _indent_sql(right_sql, indent=2)

    return f"(\n{left_indented}\n)\nUNION\n(\n{right_indented}\n)"


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


def _indent_sql(sql: str, indent: int = 4) -> str:
    """Indent each line of a SQL string by the given number of spaces."""
    pad = " " * indent
    return "\n".join(f"{pad}{line}" for line in sql.split("\n"))


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
        left_node = condition.left
        right_node = condition.right
        left = _condition_to_sql(left_node)
        right = _condition_to_sql(right_node)

        # Normalize: put simple column reference on the left side
        # Flip the comparison if left is complex (function/literal) and right is a simple variable
        if _is_complex_expr(left_node) and isinstance(right_node, VariableRefNode):
            # Flip: "EXPR >= col" → "col <= EXPR"
            left, right = right, left
            op = _flip_operator(condition.operator)
        else:
            op = condition.operator

        return f"{left} {op} {right}"

    elif isinstance(condition, LogicalConnectiveNode):
        left = _condition_to_sql(condition.left)
        right = _condition_to_sql(condition.right)
        op = condition.operator.upper()
        if op == "IMPLIES":
            return f"(NOT ({left}) OR {right})"
        return f"({left} {op} {right})"

    elif isinstance(condition, NotNode):
        operand = _condition_to_sql(condition.operand)
        if not operand:
            return ""
        return f"NOT ({operand})"

    elif isinstance(condition, VariableRefNode):
        # Strip DRC-level suffixes (_r2, _1, _2) for SQL output
        # Only strip if the result is a valid column name (non-empty, no trailing _)
        name = condition.name
        if name.endswith("_r2") and len(name) > 3:
            base = name[:-3]
            if base and not base.endswith("_"):
                name = base
        elif len(name) > 2 and name[-2] == "_" and name[-1] in "12":
            base = name[:-2]
            if base and not base.endswith("_"):
                name = base
        return name

    elif isinstance(condition, LiteralNode):
        if condition.data_type == "string":
            val = str(condition.value)
            escaped = val.replace("'", "''")
            # Use DATE literal for date-like strings
            if _is_date_string(val):
                return f"DATE '{escaped}'"
            return f"'{escaped}'"
        else:
            return str(condition.value)

    elif isinstance(condition, MembershipNode):
        return ""

    elif isinstance(condition, QuantifierNode):
        # ∃ vars (body) → EXISTS (SELECT 1 FROM table alias WHERE alias.col = outer_alias.col)
        # ∀ vars (body) → NOT EXISTS (...)

        # If body is a membership, generate correlated EXISTS with proper aliases
        if isinstance(condition.body, MembershipNode):
            table = condition.body.relation
            alias = table[0].lower()  # first letter as alias
            quantified = set(condition.variables)
            correlated = [v for v in condition.body.variables if v not in quantified]
            if correlated:
                where_parts = [f"{alias}.{v} = {v}" for v in correlated]
                where_clause = f" WHERE {' AND '.join(where_parts)}"
            else:
                where_clause = ""
            if condition.kind == "exists":
                return f"EXISTS (SELECT 1 FROM {table} {alias}{where_clause})"
            elif condition.kind == "forall":
                return f"NOT EXISTS (SELECT 1 FROM {table} {alias}{where_clause})"
            return ""

        # If body is AND with a membership inside, extract table and conditions
        if isinstance(condition.body, LogicalConnectiveNode) and condition.body.operator == "and":
            table_name = _extract_table_from_condition(condition.body)
            if table_name:
                alias = table_name[0].lower()
                membership = _find_membership(condition.body)
                quantified = set(condition.variables)
                correlated = []
                if membership:
                    correlated = [v for v in membership.variables if v not in quantified]

                body_sql = _condition_to_sql_skip_membership(condition.body)
                
                where_parts = []
                for v in correlated:
                    where_parts.append(f"{alias}.{v} = {v}")
                if body_sql:
                    where_parts.append(body_sql)
                
                where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
                
                if condition.kind == "exists":
                    return f"EXISTS (SELECT 1 FROM {table_name} {alias}{where_clause})"
                elif condition.kind == "forall":
                    return f"NOT EXISTS (SELECT 1 FROM {table_name} {alias}{where_clause})"

        # Fallback: generic body
        body_sql = _condition_to_sql(condition.body)
        if not body_sql:
            return ""
        if condition.kind == "exists":
            return f"EXISTS (SELECT 1 WHERE {body_sql})"
        elif condition.kind == "forall":
            return f"NOT EXISTS (SELECT 1 WHERE NOT ({body_sql}))"
        return body_sql

    elif isinstance(condition, FunctionCallNode):
        if condition.function == "CURRENT_DATE":
            return "CURRENT_DATE"
        elif condition.function == "DATE_SUB" and len(condition.arguments) == 2:
            base = _condition_to_sql(condition.arguments[0])
            days_str = _condition_to_sql(condition.arguments[1])
            interval = _days_to_interval(days_str)
            return f"{base} - INTERVAL '{interval}'"
        elif condition.function == "DATE_ADD" and len(condition.arguments) == 2:
            base = _condition_to_sql(condition.arguments[0])
            days_str = _condition_to_sql(condition.arguments[1])
            interval = _days_to_interval(days_str)
            return f"{base} + INTERVAL '{interval}'"
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


def _is_complex_expr(node) -> bool:
    """Check if a node is a complex expression (not a simple variable reference)."""
    from text_to_sql_planner.types.drc import FunctionCallNode, ArithmeticNode, LiteralNode
    return isinstance(node, (FunctionCallNode, ArithmeticNode, LiteralNode))


def _extract_table_from_condition(node) -> str | None:
    """Extract a table name from a MembershipNode within a condition tree."""
    if isinstance(node, MembershipNode):
        return node.relation
    if isinstance(node, LogicalConnectiveNode):
        left = _extract_table_from_condition(node.left)
        if left:
            return left
        return _extract_table_from_condition(node.right)
    return None


def _find_membership(node):
    """Find the first MembershipNode in a condition tree."""
    if isinstance(node, MembershipNode):
        return node
    if isinstance(node, LogicalConnectiveNode):
        left = _find_membership(node.left)
        if left:
            return left
        return _find_membership(node.right)
    return None


def _condition_to_sql_skip_membership(condition) -> str:
    """Convert a condition to SQL, skipping MembershipNode (returns empty for them).
    Then filter out empty parts from AND chains."""
    if isinstance(condition, LogicalConnectiveNode) and condition.operator == "and":
        left = _condition_to_sql_skip_membership(condition.left)
        right = _condition_to_sql_skip_membership(condition.right)
        parts = [p for p in [left, right] if p]
        if not parts:
            return ""
        return " AND ".join(parts)
    if isinstance(condition, MembershipNode):
        return ""
    return _condition_to_sql(condition)


def _flip_operator(op: str) -> str:
    """Flip a comparison operator (e.g., >= becomes <=)."""
    flips = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "=": "=", "!=": "!="}
    return flips.get(op, op)


def _days_to_interval(days_str: str) -> str:
    """Convert a number of days to the most readable interval unit.

    Examples:
        "10950" → "30 years"
        "365" → "1 year"
        "730" → "2 years"
        "30" → "30 days"
        "90" → "3 months"
    """
    try:
        days = int(days_str)
    except (ValueError, TypeError):
        return f"{days_str} days"

    # Check for exact year multiples (using 365 days/year)
    if days % 365 == 0:
        years = days // 365
        if years == 1:
            return "1 year"
        return f"{years} years"

    # Check for approximate month multiples (using 30 days/month)
    if days % 30 == 0 and days < 365:
        months = days // 30
        if months == 1:
            return "1 month"
        return f"{months} months"

    # Check for week multiples
    if days % 7 == 0 and days < 30:
        weeks = days // 7
        if weeks == 1:
            return "1 week"
        return f"{weeks} weeks"

    if days == 1:
        return "1 day"
    return f"{days} days"


# --- Alias helpers ---

_alias_counter: dict[str, int] = {}


def _make_alias(table_name: str, existing_aliases: set[str] | None = None) -> str:
    """Generate a unique short alias for a table name.

    Uses first letter, appending a number if there's a conflict.
    """
    base = table_name[0].lower()
    if existing_aliases is None:
        return base
    alias = base
    counter = 1
    while alias in existing_aliases:
        counter += 1
        alias = f"{base}{counter}"
    return alias


def _qualify_column(col: str, table_aliases: dict[str, str], inputs: list) -> str:
    """Qualify a column name with the appropriate table alias.

    Looks through the inputs to find which table owns this column.
    """
    # If already qualified (contains a dot), return as-is
    if "." in col:
        return col

    # Check if it's an aggregate
    for func in _AGGREGATE_FUNCS:
        if col.startswith(f"{func}(") or col.startswith(f"({func} "):
            underlying = _extract_underlying_column(col)
            qualified_underlying = _qualify_column_from_aliases(underlying, table_aliases)
            return _col_to_sql(col).replace(underlying, qualified_underlying)

    return _qualify_column_from_aliases(col, table_aliases)


def _qualify_column_from_aliases(col: str, table_aliases: dict[str, str]) -> str:
    """Qualify a column with the appropriate table alias.

    Handles suffixed variables from cartesian products/self-joins:
    - col_r2 or col_2 → use the second table's alias with the base column name
    - col_1 → use the first table's alias with the base column name
    """
    if "." in col:
        return col

    if not table_aliases:
        return col

    aliases = list(table_aliases.values())

    # Check for _r2 suffix (from join operator renaming)
    if col.endswith("_r2") and len(col) > 3:
        base_col = col[:-3]
        if base_col and not base_col.endswith("_"):
            if len(aliases) >= 2:
                return f"{aliases[1]}.{base_col}"
            return f"{aliases[0]}.{base_col}"

    # Check for _2 suffix (from cartesian product renaming)
    if len(col) > 2 and col[-2] == "_" and col[-1] == "2":
        base_col = col[:-2]
        if base_col and not base_col.endswith("_"):
            if len(aliases) >= 2:
                return f"{aliases[1]}.{base_col}"
            return f"{aliases[0]}.{base_col}"

    # Check for _1 suffix (from cartesian product renaming)
    if len(col) > 2 and col[-2] == "_" and col[-1] == "1":
        base_col = col[:-2]
        if base_col and not base_col.endswith("_"):
            return f"{aliases[0]}.{base_col}"

    # Default: use the first alias
    return f"{aliases[0]}.{col}"


def _condition_to_sql_with_aliases(condition: DRCCondition, table_aliases: dict[str, str]) -> str:
    """Convert a condition to SQL with explicit table alias qualification.

    For now, qualifies VariableRefNode with the first available alias.
    """
    if condition is None:
        return ""

    if isinstance(condition, ComparisonNode):
        left_node = condition.left
        right_node = condition.right
        left = _condition_to_sql_with_aliases(left_node, table_aliases)
        right = _condition_to_sql_with_aliases(right_node, table_aliases)

        if _is_complex_expr(left_node) and isinstance(right_node, VariableRefNode):
            left, right = right, left
            op = _flip_operator(condition.operator)
        else:
            op = condition.operator
        return f"{left} {op} {right}"

    elif isinstance(condition, LogicalConnectiveNode):
        left = _condition_to_sql_with_aliases(condition.left, table_aliases)
        right = _condition_to_sql_with_aliases(condition.right, table_aliases)
        op = condition.operator.upper()
        if op == "IMPLIES":
            return f"(NOT ({left}) OR {right})"
        parts = [p for p in [left, right] if p]
        if not parts:
            return ""
        return f"({' {0} '.format(op).join(parts)})"

    elif isinstance(condition, NotNode):
        operand = _condition_to_sql_with_aliases(condition.operand, table_aliases)
        if not operand:
            return ""
        return f"NOT ({operand})"

    elif isinstance(condition, VariableRefNode):
        return _qualify_column_from_aliases(condition.name, table_aliases)

    elif isinstance(condition, LiteralNode):
        if condition.data_type == "string":
            val = str(condition.value)
            escaped = val.replace("'", "''")
            if _is_date_string(val):
                return f"DATE '{escaped}'"
            return f"'{escaped}'"
        return str(condition.value)

    elif isinstance(condition, MembershipNode):
        return ""

    elif isinstance(condition, QuantifierNode):
        # Use the same quantifier logic but with outer aliases for correlation
        return _quantifier_to_sql_with_aliases(condition, table_aliases)

    elif isinstance(condition, FunctionCallNode):
        if condition.function == "CURRENT_DATE":
            return "CURRENT_DATE"
        elif condition.function == "DATE_SUB" and len(condition.arguments) == 2:
            base = _condition_to_sql_with_aliases(condition.arguments[0], table_aliases)
            days_str = _condition_to_sql_with_aliases(condition.arguments[1], table_aliases)
            interval = _days_to_interval(days_str)
            return f"{base} - INTERVAL '{interval}'"
        elif condition.function == "DATE_ADD" and len(condition.arguments) == 2:
            base = _condition_to_sql_with_aliases(condition.arguments[0], table_aliases)
            days_str = _condition_to_sql_with_aliases(condition.arguments[1], table_aliases)
            interval = _days_to_interval(days_str)
            return f"{base} + INTERVAL '{interval}'"
        else:
            if not condition.arguments:
                return condition.function
            args = ", ".join(_condition_to_sql_with_aliases(arg, table_aliases) for arg in condition.arguments)
            return f"{condition.function}({args})"

    return ""


def _quantifier_to_sql_with_aliases(condition: QuantifierNode, outer_aliases: dict[str, str]) -> str:
    """Convert a quantifier to SQL with explicit outer alias for correlation."""
    if isinstance(condition.body, MembershipNode):
        table = condition.body.relation
        inner_alias = _make_alias(table)
        # Avoid alias collision with outer
        if inner_alias in outer_aliases.values():
            inner_alias = inner_alias + "2"
        quantified = set(condition.variables)
        correlated = [v for v in condition.body.variables if v not in quantified]
        if correlated:
            # Find the outer alias for the correlated variable
            outer_alias = list(outer_aliases.values())[0] if outer_aliases else ""
            where_parts = [f"{inner_alias}.{v} = {outer_alias}.{v}" for v in correlated]
            where_clause = f" WHERE {' AND '.join(where_parts)}"
        else:
            where_clause = ""
        if condition.kind == "exists":
            return f"EXISTS (SELECT 1 FROM {table} {inner_alias}{where_clause})"
        elif condition.kind == "forall":
            return f"NOT EXISTS (SELECT 1 FROM {table} {inner_alias}{where_clause})"
        return ""

    # Body is AND with membership
    if isinstance(condition.body, LogicalConnectiveNode) and condition.body.operator == "and":
        table_name = _extract_table_from_condition(condition.body)
        if table_name:
            inner_alias = _make_alias(table_name)
            if inner_alias in outer_aliases.values():
                inner_alias = inner_alias + "2"
            membership = _find_membership(condition.body)
            quantified = set(condition.variables)
            correlated = []
            if membership:
                correlated = [v for v in membership.variables if v not in quantified]

            body_sql = _condition_to_sql_skip_membership(condition.body)
            outer_alias = list(outer_aliases.values())[0] if outer_aliases else ""

            where_parts = []
            for v in correlated:
                where_parts.append(f"{inner_alias}.{v} = {outer_alias}.{v}")
            if body_sql:
                where_parts.append(body_sql)

            where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

            if condition.kind == "exists":
                return f"EXISTS (SELECT 1 FROM {table_name} {inner_alias}{where_clause})"
            elif condition.kind == "forall":
                return f"NOT EXISTS (SELECT 1 FROM {table_name} {inner_alias}{where_clause})"

    return ""


def _is_date_string(s: str) -> bool:
    """Check if a string looks like a date (YYYY-MM-DD or YYYY/MM/DD)."""
    import re
    return bool(re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}$", s))
