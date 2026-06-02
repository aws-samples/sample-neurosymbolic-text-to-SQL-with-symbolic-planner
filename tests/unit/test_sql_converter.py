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



# ---------------------------------------------------------------------------
# Result-variable finalisation: avoid double-wrapping a root projection
# ---------------------------------------------------------------------------
#
# When the operation tree's root is a projection and the caller passes
# ``result_variables``, the converter must NOT wrap the projection in
# another SELECT. Doing so produces ``SELECT COUNT(s2.x) FROM (SELECT
# COUNT(x) FROM …) s2`` whose outer COUNT collapses to 1 — semantically
# wrong. The fix walks through any chain of root-level projections and
# builds the FROM context from the deepest non-projection ancestor;
# ``result_variables`` then drives a single SELECT at the top.

from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ColumnVariable,
)


def test_aggregate_count_over_root_projection_no_double_wrap():
    """Reproduces the nohup3.md "How many employees are at least 30
    years old?" failure: tree root is ``projection [(COUNT emp_id)]``
    over a ``projection [emp_id]`` over a selection.

    Before the fix the SQL was

        SELECT COUNT(s2.emp_id) FROM (
          SELECT COUNT(emp_id) FROM (
            SELECT e1.emp_id FROM Employees e1 WHERE …
          ) s1
        ) s2

    whose outer ``COUNT`` always returns 1. After the fix the projection
    chain is collapsed and the COUNT is applied once at the outer SELECT.
    """
    employees = _table_leaf(
        "Employees",
        ["emp_id", "first_name", "date_of_birth"],
    )
    cond = ComparisonNode(
        operator="<=",
        left=VariableRefNode(name="date_of_birth"),
        right=LiteralNode(value=20000, data_type="number"),
    )
    sel = _selection_node(employees, cond)
    proj_emp_id = _projection_node(sel, ["emp_id"])
    proj_count = OperatorNode(
        operator="projection",
        params=ProjectionParams(columns=["(COUNT emp_id)"]),
        inputs=[proj_emp_id],
        output_columns=["emp_id"],
    )
    tree = OperationTree(root=proj_count)

    result = convert_to_sql(
        tree,
        result_variables=[AggregateVariable(function="COUNT", column="emp_id")],
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql

    # Exactly ONE COUNT(...) — not two.
    assert sql.count("COUNT(") == 1, sql
    # No nested ``SELECT COUNT(`` inside a derived subquery.
    assert "FROM (\n" not in sql or "SELECT COUNT" not in sql.split("FROM (\n", 1)[1]
    # The aggregate references the underlying column (qualified by alias).
    assert "COUNT(e1.emp_id)" in sql or "COUNT(emp_id)" in sql


def test_plain_column_finalisation_skips_root_projection():
    """A non-aggregate result-variable list over a root-projection tree
    also collapses — the projection is redundant when the outer SELECT
    already supplies the column list."""
    employees = _table_leaf("Employees", ["emp_id", "first_name", "last_name"])
    proj = _projection_node(employees, ["emp_id", "first_name"])
    tree = OperationTree(root=proj)

    result = convert_to_sql(
        tree,
        result_variables=[
            ColumnVariable(name="emp_id"),
            ColumnVariable(name="first_name"),
        ],
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql

    # Top-level FROM is the bare ``Employees e1`` table, not a derived
    # subquery wrapping a projection.
    assert "FROM Employees" in sql
    # No derived-table syntax around a SELECT.
    assert "FROM (" not in sql


def test_aggregate_over_table_leaf_emits_single_count():
    """Sanity check: ``COUNT`` over a plain table leaf (no projection
    chain) emits a single SELECT — proves the fix doesn't regress the
    common case."""
    employees = _table_leaf("Employees", ["emp_id"])
    tree = OperationTree(root=employees)

    result = convert_to_sql(
        tree,
        result_variables=[AggregateVariable(function="COUNT", column="emp_id")],
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql
    assert sql.count("COUNT(") == 1
    assert "SELECT COUNT(" in sql
    assert "FROM (" not in sql


def test_count_over_projection_over_selection():
    """End-to-end: tree root is ``projection [(COUNT emp_id)]`` over
    a selection. Result is a single aggregated SELECT with the WHERE
    clause inline — no wrapper subquery."""
    employees = _table_leaf("Employees", ["emp_id", "date_of_birth"])
    cond = ComparisonNode(
        operator="<=",
        left=VariableRefNode(name="date_of_birth"),
        right=LiteralNode(value=20000, data_type="number"),
    )
    sel = _selection_node(employees, cond)
    proj_count = OperatorNode(
        operator="projection",
        params=ProjectionParams(columns=["(COUNT emp_id)"]),
        inputs=[sel],
        output_columns=["emp_id"],
    )
    tree = OperationTree(root=proj_count)

    result = convert_to_sql(
        tree,
        result_variables=[AggregateVariable(function="COUNT", column="emp_id")],
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql

    assert sql.count("COUNT(") == 1
    assert "FROM Employees" in sql
    assert "WHERE" in sql
    assert "FROM (" not in sql  # no wrapper subquery


def test_double_projection_chain_skipped():
    """Even a chain of multiple projections at the root collapses.
    The walk-through-projections logic must handle arbitrary depth.
    """
    employees = _table_leaf("Employees", ["emp_id", "first_name", "last_name"])
    p1 = _projection_node(employees, ["emp_id", "first_name"])
    p2 = _projection_node(p1, ["emp_id"])
    p3 = _projection_node(p2, ["emp_id"])
    tree = OperationTree(root=p3)

    result = convert_to_sql(
        tree,
        result_variables=[ColumnVariable(name="emp_id")],
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql

    assert "FROM Employees" in sql
    assert "FROM (" not in sql



# ---------------------------------------------------------------------------
# Anti-join → NOT EXISTS
# ---------------------------------------------------------------------------

from text_to_sql_planner.types.operators import AntiJoinParams, DifferenceParams, JoinParams


def test_antijoin_emits_not_exists():
    """A bare anti-join tree emits ``SELECT … FROM L WHERE NOT EXISTS
    (SELECT 1 FROM R …)`` — no derived-subquery wrap on either side
    when both sides are bare tables."""
    employees = _table_leaf("Employees", ["emp_id", "first_name", "last_name"])
    reviews = _table_leaf("Performance_Reviews", ["review_id", "emp_id"])

    anti = OperatorNode(
        operator="anti_join",
        params=AntiJoinParams(join_columns=["emp_id"]),
        inputs=[employees, reviews],
        output_columns=["emp_id", "first_name", "last_name"],
    )
    tree = OperationTree(root=anti)
    result = convert_to_sql(
        tree,
        result_variables=[
            ColumnVariable(name="emp_id"),
            ColumnVariable(name="first_name"),
            ColumnVariable(name="last_name"),
        ],
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql

    # The emitted SQL has NOT EXISTS, references both tables directly,
    # and doesn't have a derived ``FROM (SELECT … ) sN`` wrapper.
    assert "NOT EXISTS" in sql
    assert "FROM Employees" in sql
    assert "FROM Performance_Reviews" in sql
    # No derived subquery wrap (neither outer nor in the NOT EXISTS).
    assert "FROM (\n" not in sql


def test_antijoin_via_join_difference_pattern_end_to_end():
    """End-to-end: a tree shaped ``Join(Employees, Difference(π_emp_id(Employees),
    π_emp_id(Performance_Reviews)), key=emp_id)`` runs through the
    operation-tree simplifier (collapsing to AntiJoin) and the SQL
    converter (emitting NOT EXISTS) — producing the user's expected
    "employees with no performance reviews" form.
    """
    employees = _table_leaf("Employees", ["emp_id", "first_name", "last_name"])
    reviews = _table_leaf("Performance_Reviews", ["review_id", "emp_id"])

    emp_proj = OperatorNode(
        operator="projection",
        params=ProjectionParams(columns=["emp_id"]),
        inputs=[employees],
        output_columns=["emp_id"],
    )
    rev_proj = OperatorNode(
        operator="projection",
        params=ProjectionParams(columns=["emp_id"]),
        inputs=[reviews],
        output_columns=["emp_id"],
    )
    diff = OperatorNode(
        operator="difference",
        params=DifferenceParams(),
        inputs=[emp_proj, rev_proj],
        output_columns=["emp_id"],
    )
    join = OperatorNode(
        operator="join",
        params=JoinParams(join_columns=["emp_id"]),
        inputs=[employees, diff],
        output_columns=["emp_id", "first_name", "last_name"],
    )
    tree = OperationTree(root=join)

    result = convert_to_sql(
        tree,
        result_variables=[
            ColumnVariable(name="emp_id"),
            ColumnVariable(name="first_name"),
            ColumnVariable(name="last_name"),
        ],
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql

    assert "NOT EXISTS" in sql
    # No EXCEPT — the difference operator was eliminated.
    assert "EXCEPT" not in sql
    assert "FROM Employees" in sql
    assert "FROM Performance_Reviews" in sql
    # No derived subqueries.
    assert "FROM (\n" not in sql



def test_difference_join_pattern_end_to_end():
    """End-to-end: ``Difference(Employees, π_cols(Employees ⋈
    Performance_Reviews))`` runs through the operation-tree
    simplifier (collapsing to AntiJoin via the difference-join pass)
    and the SQL converter (emitting NOT EXISTS).
    """
    employees = _table_leaf(
        "Employees", ["emp_id", "first_name", "last_name"]
    )
    reviews = _table_leaf(
        "Performance_Reviews", ["review_id", "emp_id"]
    )

    join_inner = OperatorNode(
        operator="join",
        params=JoinParams(join_columns=["emp_id"]),
        inputs=[employees, reviews],
        output_columns=[
            "emp_id", "first_name", "last_name", "review_id",
        ],
    )
    join_proj = OperatorNode(
        operator="projection",
        params=ProjectionParams(
            columns=["emp_id", "first_name", "last_name"],
        ),
        inputs=[join_inner],
        output_columns=["emp_id", "first_name", "last_name"],
    )
    diff = OperatorNode(
        operator="difference",
        params=DifferenceParams(),
        inputs=[employees, join_proj],
        output_columns=["emp_id", "first_name", "last_name"],
    )
    tree = OperationTree(root=diff)

    result = convert_to_sql(
        tree,
        result_variables=[
            ColumnVariable(name="emp_id"),
            ColumnVariable(name="first_name"),
            ColumnVariable(name="last_name"),
        ],
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql

    assert "NOT EXISTS" in sql
    assert "EXCEPT" not in sql
    assert "FROM Employees" in sql
    assert "FROM Performance_Reviews" in sql
    # No derived subqueries.
    assert "FROM (\n" not in sql



def test_three_way_join_difference_pattern_end_to_end():
    """End-to-end: ``Join(Employees, Difference(π_emp_id(Training_Enrollment),
    π_emp_id(Performance_Reviews)))`` with DISTINCT.

    Should produce ``SELECT DISTINCT ... FROM Employees JOIN
    Training_Enrollment ... WHERE NOT EXISTS (... Performance_Reviews
    ...)``.
    """
    employees = _table_leaf(
        "Employees", ["emp_id", "first_name", "last_name"]
    )
    training = _table_leaf(
        "Training_Enrollment", ["enrollment_id", "emp_id"]
    )
    reviews = _table_leaf(
        "Performance_Reviews", ["review_id", "emp_id"]
    )

    te_proj = OperatorNode(
        operator="projection",
        params=ProjectionParams(columns=["emp_id"]),
        inputs=[training],
        output_columns=["emp_id"],
    )
    pr_proj = OperatorNode(
        operator="projection",
        params=ProjectionParams(columns=["emp_id"]),
        inputs=[reviews],
        output_columns=["emp_id"],
    )
    diff = OperatorNode(
        operator="difference",
        params=DifferenceParams(),
        inputs=[te_proj, pr_proj],
        output_columns=["emp_id"],
    )
    join = OperatorNode(
        operator="join",
        params=JoinParams(join_columns=["emp_id"]),
        inputs=[employees, diff],
        output_columns=["emp_id", "first_name", "last_name"],
    )
    tree = OperationTree(root=join)

    result = convert_to_sql(
        tree,
        result_variables=[
            ColumnVariable(name="emp_id"),
            ColumnVariable(name="first_name"),
            ColumnVariable(name="last_name"),
        ],
        distinct=True,
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql

    # NOT EXISTS for the anti-join half.
    assert "NOT EXISTS" in sql
    # Regular JOIN for the semi-join half.
    assert "JOIN Training_Enrollment" in sql or "JOIN (\n    SELECT" in sql
    assert "FROM Employees" in sql
    assert "Performance_Reviews" in sql
    # DISTINCT at the top level.
    assert "DISTINCT" in sql
    # No EXCEPT.
    assert "EXCEPT" not in sql



# ---------------------------------------------------------------------------
# Rename operator
# ---------------------------------------------------------------------------

from text_to_sql_planner.types.operators import RenameParams


def _rename_node(input_node, mapping: dict[str, str]) -> OperatorNode:
    """Construct a rename operator node with derived output_columns."""
    base_cols = (
        list(input_node.columns)
        if isinstance(input_node, TableLeafNode)
        else list(input_node.output_columns)
    )
    new_cols = [mapping.get(c, c) for c in base_cols]
    return OperatorNode(
        operator="rename",
        params=RenameParams(mapping=dict(mapping)),
        inputs=[input_node],
        output_columns=new_cols,
    )


def test_rename_flattens_into_inner_select():
    """``ρ_{a → x}(R)`` flattens — the SQL just references ``alias.a``
    where the outer query asked for ``x``. No subquery wrapper."""
    r = _table_leaf("R", ["a", "b"])
    renamed = _rename_node(r, {"a": "x"})
    tree = OperationTree(root=renamed)

    result = convert_to_sql(
        tree,
        result_variables=[
            ColumnVariable(name="x"),
            ColumnVariable(name="b"),
        ],
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql
    # The SELECT projects x AS x (the inner alias's a column under
    # the new name), and b unchanged.
    assert "FROM R" in sql
    # Inner alias.column reference is preserved (no derived subquery).
    assert "FROM (\n" not in sql


def test_rename_used_for_self_join_disambiguation():
    """Demonstrates the canonical use case: self-join two copies of
    the same relation, each bound to a different alias via rename so
    the natural join doesn't collapse on shared columns.

    Tree: Join( Performance_Reviews_a,
                ρ_{review_id ← review_id_2}(Performance_Reviews_b),
                key=emp_id )

    The SQL output should join the two PR instances on emp_id, with
    one copy's ``review_id`` accessible as ``review_id_2`` so a later
    selection can require ``review_id != review_id_2`` (or whatever).
    """
    pr_a = _table_leaf(
        "Performance_Reviews", ["review_id", "emp_id"],
    )
    pr_b = _table_leaf(
        "Performance_Reviews", ["review_id", "emp_id"],
    )
    pr_b_renamed = _rename_node(pr_b, {"review_id": "review_id_2"})

    join = OperatorNode(
        operator="join",
        params=JoinParams(join_columns=["emp_id"]),
        inputs=[pr_a, pr_b_renamed],
        output_columns=["review_id", "emp_id", "review_id_2"],
    )
    tree = OperationTree(root=join)

    result = convert_to_sql(
        tree,
        result_variables=[
            ColumnVariable(name="review_id"),
            ColumnVariable(name="emp_id"),
            ColumnVariable(name="review_id_2"),
        ],
    )
    assert isinstance(result, SQLSuccess), result
    sql = result.sql

    # Both Performance_Reviews instances appear in the FROM/JOIN
    # (they get separate aliases automatically).
    assert sql.count("Performance_Reviews") == 2
    # No derived subquery wrapping the rename — it flattens.
    assert "FROM (\n" not in sql
