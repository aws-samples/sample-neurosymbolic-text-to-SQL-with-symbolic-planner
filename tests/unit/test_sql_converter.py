"""Unit tests for the SQL converter."""

from __future__ import annotations

import pytest

from text_to_sql_planner.sql import convert_to_sql, SQLSuccess, SQLFailure
from text_to_sql_planner.types.operation_tree import (
    OperationTree,
    TableLeafNode,
    OperatorNode,
)
from text_to_sql_planner.types.operators import (
    SelectionParams,
    JoinParams,
    ProjectionParams,
    CartesianProductParams,
    UnionParams,
    DifferenceParams,
    DivisionParams,
)
from text_to_sql_planner.types.drc import (
    ComparisonNode,
    LogicalConnectiveNode,
    NotNode,
    VariableRefNode,
    LiteralNode,
)


# --- Helpers ---


def _table_leaf(name: str, columns: list[str]) -> TableLeafNode:
    """Create a simple table leaf node."""
    return TableLeafNode(table_name=name, columns=columns)


def _selection_node(
    input_node, condition, output_columns: list[str] | None = None
) -> OperatorNode:
    """Create a selection operator node."""
    cols = output_columns or _get_cols(input_node)
    return OperatorNode(
        operator="selection",
        params=SelectionParams(condition=condition),
        inputs=[input_node],
        output_columns=cols,
    )


def _projection_node(input_node, columns: list[str]) -> OperatorNode:
    """Create a projection operator node."""
    return OperatorNode(
        operator="projection",
        params=ProjectionParams(columns=columns),
        inputs=[input_node],
        output_columns=columns,
    )


def _join_node(
    left, right, join_columns: list[str], output_columns: list[str] | None = None
) -> OperatorNode:
    """Create a join operator node."""
    if output_columns is None:
        left_cols = _get_cols(left)
        right_cols = _get_cols(right)
        output_columns = left_cols + [c for c in right_cols if c not in left_cols]
    return OperatorNode(
        operator="join",
        params=JoinParams(join_columns=join_columns),
        inputs=[left, right],
        output_columns=output_columns,
    )


def _cartesian_node(
    left, right, output_columns: list[str] | None = None
) -> OperatorNode:
    """Create a cartesian product operator node."""
    if output_columns is None:
        output_columns = _get_cols(left) + _get_cols(right)
    return OperatorNode(
        operator="cartesian_product",
        params=CartesianProductParams(),
        inputs=[left, right],
        output_columns=output_columns,
    )


def _union_node(left, right, output_columns: list[str] | None = None) -> OperatorNode:
    """Create a union operator node."""
    if output_columns is None:
        output_columns = _get_cols(left)
    return OperatorNode(
        operator="union",
        params=UnionParams(),
        inputs=[left, right],
        output_columns=output_columns,
    )


def _difference_node(left, right, output_columns: list[str] | None = None) -> OperatorNode:
    """Create a set-difference operator node."""
    if output_columns is None:
        output_columns = _get_cols(left)
    return OperatorNode(
        operator="difference",
        params=DifferenceParams(),
        inputs=[left, right],
        output_columns=output_columns,
    )


def _division_node(
    left, right, output_columns: list[str] | None = None
) -> OperatorNode:
    """Create a division operator node."""
    if output_columns is None:
        left_cols = _get_cols(left)
        right_cols = _get_cols(right)
        output_columns = [c for c in left_cols if c not in right_cols]
    return OperatorNode(
        operator="division",
        params=DivisionParams(),
        inputs=[left, right],
        output_columns=output_columns,
    )


def _get_cols(node) -> list[str]:
    """Get columns from a node."""
    if isinstance(node, TableLeafNode):
        return node.columns
    elif isinstance(node, OperatorNode):
        return node.output_columns
    return []


# --- Table Leaf Tests ---


class TestTableLeaf:
    def test_simple_table(self):
        """Table leaf produces SELECT alias.columns FROM table alias."""
        tree = OperationTree(root=_table_leaf("employees", ["id", "name"]))
        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        # Uses numbered aliases: e1.id, e1.name, FROM employees e1
        assert ".id" in result.sql
        assert ".name" in result.sql
        assert "FROM employees e" in result.sql

    def test_table_with_no_columns(self):
        """Table leaf with no columns produces SELECT * FROM table alias."""
        tree = OperationTree(root=_table_leaf("orders", []))
        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "SELECT *" in result.sql
        assert "FROM orders" in result.sql


# --- Selection Tests ---


class TestSelection:
    def test_simple_where(self):
        """Selection produces a WHERE clause."""
        table = _table_leaf("employees", ["id", "name", "age"])
        condition = ComparisonNode(
            operator=">",
            left=VariableRefNode(name="age"),
            right=LiteralNode(value=30, data_type="number"),
        )
        node = _selection_node(table, condition)
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "FROM employees" in result.sql
        assert "WHERE" in result.sql
        assert "age > 30" in result.sql

    def test_string_literal_in_condition(self):
        """Selection with string literal uses quoted value."""
        table = _table_leaf("employees", ["id", "name"])
        condition = ComparisonNode(
            operator="=",
            left=VariableRefNode(name="name"),
            right=LiteralNode(value="Alice", data_type="string"),
        )
        node = _selection_node(table, condition)
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "'Alice'" in result.sql

    def test_logical_and_condition(self):
        """Selection with AND condition."""
        table = _table_leaf("employees", ["id", "name", "age"])
        condition = LogicalConnectiveNode(
            operator="and",
            left=ComparisonNode(
                operator=">",
                left=VariableRefNode(name="age"),
                right=LiteralNode(value=25, data_type="number"),
            ),
            right=ComparisonNode(
                operator="<",
                left=VariableRefNode(name="age"),
                right=LiteralNode(value=65, data_type="number"),
            ),
        )
        node = _selection_node(table, condition)
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "AND" in result.sql
        assert "age > 25" in result.sql
        assert "age < 65" in result.sql

    def test_not_condition(self):
        """Selection with NOT condition."""
        table = _table_leaf("employees", ["id", "active"])
        condition = NotNode(
            operand=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="active"),
                right=LiteralNode(value=0, data_type="number"),
            )
        )
        node = _selection_node(table, condition)
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "NOT" in result.sql
        assert "active = 0" in result.sql


# --- Projection Tests ---


class TestProjection:
    def test_select_specific_columns(self):
        """Projection produces SELECT with specific columns."""
        table = _table_leaf("employees", ["id", "name", "age"])
        node = _projection_node(table, ["name", "age"])
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        # Columns are always qualified with the table alias to avoid
        # ambiguity in the presence of joins.
        assert "SELECT e1.name, e1.age" in result.sql
        assert "FROM employees" in result.sql

    def test_single_column(self):
        """Projection with a single column."""
        table = _table_leaf("employees", ["id", "name", "age"])
        node = _projection_node(table, ["id"])
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "SELECT e1.id" in result.sql
        assert "FROM employees" in result.sql

    def test_error_invalid_column(self):
        """Projection fails when column doesn't exist in input."""
        table = _table_leaf("employees", ["id", "name"])
        node = _projection_node(table, ["id", "salary"])
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLFailure)
        assert "salary" in result.error


# --- Join Tests ---


class TestJoin:
    def test_simple_join(self):
        """Join produces JOIN ON clause with aliases."""
        left = _table_leaf("employees", ["id", "name"])
        right = _table_leaf("departments", ["id", "dept_name"])
        node = _join_node(left, right, ["id"])
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "JOIN" in result.sql
        assert "ON" in result.sql
        # Should use aliases like e.id = d.id
        assert ".id" in result.sql

    def test_join_with_multiple_columns(self):
        """Join with multiple join columns."""
        left = _table_leaf("orders", ["customer_id", "product_id", "qty"])
        right = _table_leaf("prices", ["customer_id", "product_id", "price"])
        node = _join_node(left, right, ["customer_id", "product_id"])
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "customer_id" in result.sql
        assert "product_id" in result.sql
        assert "ON" in result.sql

    def test_join_output_columns(self):
        """Join includes output columns in SELECT."""
        left = _table_leaf("employees", ["id", "name"])
        right = _table_leaf("departments", ["id", "dept_name"])
        node = _join_node(left, right, ["id"], output_columns=["id", "name", "dept_name"])
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "SELECT" in result.sql
        assert "id" in result.sql
        assert "name" in result.sql
        assert "dept_name" in result.sql


# --- Cartesian Product Tests ---


class TestCartesianProduct:
    def test_cross_join(self):
        """Cartesian product produces CROSS JOIN."""
        left = _table_leaf("colors", ["color"])
        right = _table_leaf("sizes", ["size"])
        node = _cartesian_node(left, right)
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "CROSS JOIN" in result.sql
        assert "colors" in result.sql
        assert "sizes" in result.sql

    def test_cross_join_output_columns(self):
        """Cartesian product includes output columns."""
        left = _table_leaf("t1", ["a", "b"])
        right = _table_leaf("t2", ["c", "d"])
        node = _cartesian_node(left, right, output_columns=["a", "b", "c", "d"])
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "CROSS JOIN" in result.sql
        # Should have qualified columns
        assert ".a" in result.sql or "a" in result.sql


# --- Union Tests ---


class TestUnion:
    def test_simple_union(self):
        """Union produces UNION of two SELECTs."""
        left = _table_leaf("employees", ["id", "name"])
        right = _table_leaf("contractors", ["id", "name"])
        node = _union_node(left, right)
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "UNION" in result.sql
        assert "employees" in result.sql
        assert "contractors" in result.sql

    def test_union_wraps_tables_as_selects(self):
        """Union wraps table references as full SELECT statements."""
        left = _table_leaf("t1", ["id", "name"])
        right = _table_leaf("t2", ["id", "name"])
        node = _union_node(left, right)
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "SELECT" in result.sql
        assert "FROM t1" in result.sql
        assert "FROM t2" in result.sql


class TestDifference:
    def test_simple_difference(self):
        """Difference produces EXCEPT of two SELECTs."""
        left = _table_leaf("employees", ["id", "name"])
        right = _table_leaf("contractors", ["id", "name"])
        node = _difference_node(left, right)
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "EXCEPT" in result.sql
        assert "employees" in result.sql
        assert "contractors" in result.sql

    def test_difference_wraps_tables_as_selects(self):
        """Difference wraps table references as full SELECT statements."""
        left = _table_leaf("t1", ["id", "name"])
        right = _table_leaf("t2", ["id", "name"])
        node = _difference_node(left, right)
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "EXCEPT" in result.sql
        assert "FROM t1" in result.sql
        assert "FROM t2" in result.sql


# --- Nested Operations Tests ---


class TestNestedOperations:
    def test_selection_then_projection(self):
        """Nested: project columns from a filtered table."""
        table = _table_leaf("employees", ["id", "name", "age"])
        selection = _selection_node(
            table,
            ComparisonNode(
                operator=">",
                left=VariableRefNode(name="age"),
                right=LiteralNode(value=30, data_type="number"),
            ),
            output_columns=["id", "name", "age"],
        )
        projection = _projection_node(selection, ["name", "age"])
        tree = OperationTree(root=projection)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        # Projected columns are always qualified to avoid ambiguity.
        assert "SELECT e1.name, e1.age" in result.sql
        assert "WHERE" in result.sql
        assert "age > 30" in result.sql

    def test_join_then_selection(self):
        """Nested: filter results of a join."""
        left = _table_leaf("employees", ["id", "name", "dept_id"])
        right = _table_leaf("departments", ["dept_id", "dept_name"])
        join = _join_node(left, right, ["dept_id"],
                          output_columns=["id", "name", "dept_id", "dept_name"])
        selection = _selection_node(
            join,
            ComparisonNode(
                operator="=",
                left=VariableRefNode(name="dept_name"),
                right=LiteralNode(value="Engineering", data_type="string"),
            ),
            output_columns=["id", "name", "dept_id", "dept_name"],
        )
        tree = OperationTree(root=selection)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "JOIN" in result.sql
        assert "WHERE" in result.sql
        assert "'Engineering'" in result.sql

    def test_projection_after_cartesian(self):
        """Nested: project columns from a cartesian product."""
        left = _table_leaf("colors", ["color"])
        right = _table_leaf("sizes", ["size"])
        cartesian = _cartesian_node(left, right, output_columns=["color", "size"])
        projection = _projection_node(cartesian, ["color"])
        tree = OperationTree(root=projection)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "SELECT c1.color" in result.sql
        assert "CROSS JOIN" in result.sql


# --- Error Cases ---


class TestErrors:
    def test_none_tree(self):
        """Error: None tree."""
        result = convert_to_sql(None)  # type: ignore

        assert isinstance(result, SQLFailure)
        assert "None" in result.error

    def test_none_root(self):
        """Error: tree with None root."""
        tree = OperationTree(root=None)
        result = convert_to_sql(tree)

        assert isinstance(result, SQLFailure)
        assert "None" in result.error

    def test_empty_table_name(self):
        """Error: table leaf with empty name."""
        tree = OperationTree(root=TableLeafNode(table_name="", columns=["id"]))
        result = convert_to_sql(tree)

        assert isinstance(result, SQLFailure)
        assert "empty" in result.error.lower()

    def test_selection_no_condition(self):
        """Error: selection with no condition."""
        table = _table_leaf("t", ["id"])
        node = OperatorNode(
            operator="selection",
            params=SelectionParams(condition=None),
            inputs=[table],
            output_columns=["id"],
        )
        tree = OperationTree(root=node)
        result = convert_to_sql(tree)

        assert isinstance(result, SQLFailure)
        assert "condition" in result.error.lower()

    def test_projection_invalid_column(self):
        """Error: projection references non-existent column."""
        table = _table_leaf("employees", ["id", "name"])
        node = _projection_node(table, ["id", "nonexistent"])
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLFailure)
        assert "nonexistent" in result.error

    def test_join_no_inputs(self):
        """Error: join with no inputs."""
        node = OperatorNode(
            operator="join",
            params=JoinParams(join_columns=["id"]),
            inputs=[],
            output_columns=["id"],
        )
        tree = OperationTree(root=node)
        result = convert_to_sql(tree)

        assert isinstance(result, SQLFailure)
        assert "2 inputs" in result.error or "requires" in result.error.lower()

    def test_union_single_input(self):
        """Error: union with only one input."""
        table = _table_leaf("t1", ["id"])
        node = OperatorNode(
            operator="union",
            params=UnionParams(),
            inputs=[table],
            output_columns=["id"],
        )
        tree = OperationTree(root=node)
        result = convert_to_sql(tree)

        assert isinstance(result, SQLFailure)
        assert "2 inputs" in result.error or "requires" in result.error.lower()


# --- Division Tests ---


class TestDivision:
    def test_simple_division(self):
        """Division produces double NOT EXISTS pattern."""
        left = _table_leaf("enrollments", ["student", "course"])
        right = _table_leaf("required_courses", ["course"])
        node = _division_node(left, right)
        tree = OperationTree(root=node)

        result = convert_to_sql(tree)

        assert isinstance(result, SQLSuccess)
        assert "NOT EXISTS" in result.sql
        assert "student" in result.sql
        # Should have nested NOT EXISTS
        assert result.sql.count("NOT EXISTS") == 2
