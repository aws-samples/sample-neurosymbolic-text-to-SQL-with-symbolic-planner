"""Unit tests for relational algebra operators and dispatcher."""

from __future__ import annotations

import pytest

from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    MembershipNode,
    LogicalConnectiveNode,
    ComparisonNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
    LiteralNode,
)
from text_to_sql_planner.types.operators import (
    OperatorApplication,
    OperatorSuccess,
    OperatorFailure,
    SelectionParams,
    JoinParams,
    ProjectionParams,
    CartesianProductParams,
    UnionParams,
    DifferenceParams,
    DivisionParams,
)
from text_to_sql_planner.operators import apply_operator
from text_to_sql_planner.operators.selection import apply_selection
from text_to_sql_planner.operators.join import apply_join
from text_to_sql_planner.operators.projection import apply_projection
from text_to_sql_planner.operators.cartesian_product import apply_cartesian_product
from text_to_sql_planner.operators.union import apply_union
from text_to_sql_planner.operators.difference import apply_difference
from text_to_sql_planner.operators.division import apply_division


# --- Helpers ---

def _make_relation(columns: list[str], relation_name: str) -> DRCExpression:
    """Create a simple DRC expression representing a base relation."""
    result_variables = [ColumnVariable(name=col) for col in columns]
    condition = MembershipNode(variables=list(columns), relation=relation_name)
    return DRCExpression(result_variables=result_variables, condition=condition)


def _get_column_names(expr: DRCExpression) -> list[str]:
    """Extract column names from a DRC expression."""
    return [
        rv.name if isinstance(rv, ColumnVariable) else rv.column
        for rv in expr.result_variables
    ]


# --- Selection Tests ---

class TestSelection:
    def test_basic_filter(self):
        """Selection applies a condition to the input relation."""
        relation = _make_relation(["id", "name", "age"], "employees")
        condition = ComparisonNode(
            operator=">",
            left=VariableRefNode(name="age"),
            right=LiteralNode(value=30, data_type="number"),
        )
        params = SelectionParams(condition=condition)

        result = apply_selection(params, [relation])

        assert isinstance(result, OperatorSuccess)
        output = result.output
        # Output condition should be (and original_condition selection_condition)
        assert isinstance(output.condition, LogicalConnectiveNode)
        assert output.condition.operator == "and"
        assert output.condition.left == relation.condition
        assert output.condition.right == condition

    def test_output_columns_unchanged(self):
        """Selection preserves all columns from the input."""
        relation = _make_relation(["id", "name", "age"], "employees")
        condition = ComparisonNode(
            operator="=",
            left=VariableRefNode(name="name"),
            right=LiteralNode(value="Alice", data_type="string"),
        )
        params = SelectionParams(condition=condition)

        result = apply_selection(params, [relation])

        assert isinstance(result, OperatorSuccess)
        assert _get_column_names(result.output) == ["id", "name", "age"]

    def test_error_no_inputs(self):
        """Selection fails with no input relations."""
        condition = ComparisonNode(
            operator="=",
            left=VariableRefNode(name="x"),
            right=LiteralNode(value=1, data_type="number"),
        )
        params = SelectionParams(condition=condition)

        result = apply_selection(params, [])

        assert isinstance(result, OperatorFailure)
        assert "1 input" in result.error

    def test_error_no_condition(self):
        """Selection fails when no condition is provided."""
        relation = _make_relation(["id"], "t")
        params = SelectionParams(condition=None)

        result = apply_selection(params, [relation])

        assert isinstance(result, OperatorFailure)
        assert "condition" in result.error.lower()


# --- Join Tests ---

class TestJoin:
    def test_valid_join(self):
        """Join produces correct output for valid inputs."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["id", "dept"], "departments")
        params = JoinParams(join_columns=["id"])

        result = apply_join(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        output = result.output
        # Output should have columns from both, with join column once
        col_names = _get_column_names(output)
        assert "id" in col_names
        assert "name" in col_names
        assert "dept" in col_names

    def test_join_column_appears_once(self):
        """Join column should appear only once in the output."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["id", "salary"], "salaries")
        params = JoinParams(join_columns=["id"])

        result = apply_join(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        col_names = _get_column_names(result.output)
        assert col_names.count("id") == 1
        assert len(col_names) == 3  # id, name, salary

    def test_error_nonexistent_column_in_first(self):
        """Join fails when join column doesn't exist in first relation."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["emp_id", "dept"], "departments")
        params = JoinParams(join_columns=["emp_id"])

        result = apply_join(params, [r1, r2])

        assert isinstance(result, OperatorFailure)
        assert "emp_id" in result.error
        assert "first" in result.error.lower()

    def test_error_nonexistent_column_in_second(self):
        """Join fails when join column doesn't exist in second relation."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["dept_id", "dept"], "departments")
        params = JoinParams(join_columns=["id"])

        result = apply_join(params, [r1, r2])

        assert isinstance(result, OperatorFailure)
        assert "id" in result.error
        assert "second" in result.error.lower()

    def test_error_no_join_columns(self):
        """Join fails when no join columns are specified."""
        r1 = _make_relation(["id"], "t1")
        r2 = _make_relation(["id"], "t2")
        params = JoinParams(join_columns=[])

        result = apply_join(params, [r1, r2])

        assert isinstance(result, OperatorFailure)

    def test_output_condition_structure(self):
        """Join output condition is conjunction of both input conditions."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["id", "dept"], "departments")
        params = JoinParams(join_columns=["id"])

        result = apply_join(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        # Top-level should be a logical AND of both conditions
        cond = result.output.condition
        assert isinstance(cond, LogicalConnectiveNode)
        assert cond.operator == "and"


# --- Projection Tests ---

class TestProjection:
    def test_valid_projection(self):
        """Projection selects only specified columns."""
        relation = _make_relation(["id", "name", "age"], "employees")
        params = ProjectionParams(columns=["name", "age"])

        result = apply_projection(params, [relation])

        assert isinstance(result, OperatorSuccess)
        col_names = _get_column_names(result.output)
        assert col_names == ["name", "age"]

    def test_projection_all_columns(self):
        """Projecting all columns preserves the original condition."""
        relation = _make_relation(["id", "name"], "employees")
        params = ProjectionParams(columns=["id", "name"])

        result = apply_projection(params, [relation])

        assert isinstance(result, OperatorSuccess)
        col_names = _get_column_names(result.output)
        assert col_names == ["id", "name"]
        # When no columns are removed, condition stays the same
        assert result.output.condition == relation.condition

    def test_projection_wraps_with_exists(self):
        """Projection wraps with exists quantifier binding only removed columns."""
        relation = _make_relation(["id", "name", "age"], "employees")
        params = ProjectionParams(columns=["name"])

        result = apply_projection(params, [relation])

        assert isinstance(result, OperatorSuccess)
        # Condition should be wrapped with exists for removed columns only
        cond = result.output.condition
        assert isinstance(cond, QuantifierNode)
        assert cond.kind == "exists"
        # Only removed columns (id, age) should be quantified — NOT the result variable (name)
        assert set(cond.variables) == {"id", "age"}

    def test_error_nonexistent_column(self):
        """Projection fails when a column doesn't exist in input."""
        relation = _make_relation(["id", "name"], "employees")
        params = ProjectionParams(columns=["id", "salary"])

        result = apply_projection(params, [relation])

        assert isinstance(result, OperatorFailure)
        assert "salary" in result.error

    def test_error_empty_columns(self):
        """Projection fails with empty column list."""
        relation = _make_relation(["id", "name"], "employees")
        params = ProjectionParams(columns=[])

        result = apply_projection(params, [relation])

        assert isinstance(result, OperatorFailure)


# --- Cartesian Product Tests ---

class TestCartesianProduct:
    def test_columns_are_concatenation(self):
        """Cartesian product output columns are concatenation of both inputs."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["dept_id", "dept_name"], "departments")
        params = CartesianProductParams()

        result = apply_cartesian_product(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        col_names = _get_column_names(result.output)
        assert col_names == ["id", "name", "dept_id", "dept_name"]

    def test_condition_is_conjunction(self):
        """Cartesian product condition is (and r1_condition r2_condition)."""
        r1 = _make_relation(["a"], "t1")
        r2 = _make_relation(["b"], "t2")
        params = CartesianProductParams()

        result = apply_cartesian_product(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        cond = result.output.condition
        assert isinstance(cond, LogicalConnectiveNode)
        assert cond.operator == "and"
        assert cond.left == r1.condition
        assert cond.right == r2.condition

    def test_error_wrong_input_count(self):
        """Cartesian product fails with wrong number of inputs."""
        r1 = _make_relation(["a"], "t1")
        params = CartesianProductParams()

        result = apply_cartesian_product(params, [r1])

        assert isinstance(result, OperatorFailure)
        assert "2 input" in result.error


# --- Union Tests ---

class TestUnion:
    def test_valid_union(self):
        """Union produces correct output for compatible relations."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["id", "name"], "contractors")
        params = UnionParams()

        result = apply_union(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        col_names = _get_column_names(result.output)
        assert col_names == ["id", "name"]

    def test_union_condition_is_disjunction(self):
        """Union condition is (or r1_condition r2_condition)."""
        r1 = _make_relation(["id"], "t1")
        r2 = _make_relation(["id"], "t2")
        params = UnionParams()

        result = apply_union(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        cond = result.output.condition
        assert isinstance(cond, LogicalConnectiveNode)
        assert cond.operator == "or"
        assert cond.left == r1.condition
        assert cond.right == r2.condition

    def test_error_mismatched_arity(self):
        """Union fails when relations have different number of columns."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["id"], "contractors")
        params = UnionParams()

        result = apply_union(params, [r1, r2])

        assert isinstance(result, OperatorFailure)
        assert "arity" in result.error.lower() or "2" in result.error

    def test_error_wrong_input_count(self):
        """Union fails with wrong number of inputs."""
        r1 = _make_relation(["id"], "t1")
        params = UnionParams()

        result = apply_union(params, [r1])

        assert isinstance(result, OperatorFailure)


# --- Difference Tests ---

class TestDifference:
    def test_valid_difference(self):
        """Difference produces correct columns for compatible relations."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["id", "name"], "contractors")
        params = DifferenceParams()

        result = apply_difference(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        assert _get_column_names(result.output) == ["id", "name"]

    def test_difference_condition_is_and_not(self):
        """Difference condition is (and r1_condition (not r2_condition))."""
        r1 = _make_relation(["id"], "t1")
        r2 = _make_relation(["id"], "t2")
        params = DifferenceParams()

        result = apply_difference(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        cond = result.output.condition
        assert isinstance(cond, LogicalConnectiveNode)
        assert cond.operator == "and"
        assert cond.left == r1.condition
        assert isinstance(cond.right, NotNode)
        assert cond.right.operand == r2.condition

    def test_difference_alpha_renames_right_side(self):
        """When right-side column names differ, they get rewritten to the
        left-side names so the negated condition speaks about R's tuples."""
        r1 = _make_relation(["a", "b"], "t1")
        r2 = _make_relation(["x", "y"], "t2")
        params = DifferenceParams()

        result = apply_difference(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        # The negated right side's MembershipNode should reference (a, b),
        # not (x, y).
        cond = result.output.condition
        assert isinstance(cond.right, NotNode)
        inner = cond.right.operand
        assert isinstance(inner, MembershipNode)
        assert inner.variables == ["a", "b"]
        assert inner.relation == "t2"

    def test_error_mismatched_arity(self):
        """Difference fails when relations have different number of columns."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["id"], "contractors")
        params = DifferenceParams()

        result = apply_difference(params, [r1, r2])

        assert isinstance(result, OperatorFailure)
        assert "arity" in result.error.lower()

    def test_error_wrong_input_count(self):
        """Difference fails with wrong number of inputs."""
        r1 = _make_relation(["id"], "t1")
        params = DifferenceParams()

        result = apply_difference(params, [r1])

        assert isinstance(result, OperatorFailure)


# --- Division Tests ---

class TestDivision:
    def test_valid_division(self):
        """Division produces correct output columns."""
        r1 = _make_relation(["student", "course"], "enrollments")
        r2 = _make_relation(["course"], "required_courses")
        params = DivisionParams()

        result = apply_division(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        col_names = _get_column_names(result.output)
        assert col_names == ["student"]

    def test_division_uses_forall(self):
        """Division output condition uses forall quantifier."""
        r1 = _make_relation(["student", "course"], "enrollments")
        r2 = _make_relation(["course"], "required_courses")
        params = DivisionParams()

        result = apply_division(params, [r1, r2])

        assert isinstance(result, OperatorSuccess)
        cond = result.output.condition
        assert isinstance(cond, QuantifierNode)
        assert cond.kind == "forall"
        assert "course" in cond.variables

    def test_error_columns_not_subset(self):
        """Division fails when second relation's columns aren't a subset of first."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["dept"], "departments")
        params = DivisionParams()

        result = apply_division(params, [r1, r2])

        assert isinstance(result, OperatorFailure)
        assert "dept" in result.error
        assert "subset" in result.error.lower()

    def test_error_wrong_input_count(self):
        """Division fails with wrong number of inputs."""
        r1 = _make_relation(["a", "b"], "t1")
        params = DivisionParams()

        result = apply_division(params, [r1])

        assert isinstance(result, OperatorFailure)


# --- Dispatcher Tests ---

class TestDispatcher:
    def test_routes_selection(self):
        """Dispatcher routes selection operator correctly."""
        relation = _make_relation(["id", "name"], "employees")
        condition = ComparisonNode(
            operator="=",
            left=VariableRefNode(name="id"),
            right=LiteralNode(value=1, data_type="number"),
        )
        app = OperatorApplication(
            operator="selection",
            inputs=[relation],
            params=SelectionParams(condition=condition),
        )

        result = apply_operator(app)

        assert isinstance(result, OperatorSuccess)
        assert _get_column_names(result.output) == ["id", "name"]

    def test_routes_join(self):
        """Dispatcher routes join operator correctly."""
        r1 = _make_relation(["id", "name"], "employees")
        r2 = _make_relation(["id", "dept"], "departments")
        app = OperatorApplication(
            operator="join",
            inputs=[r1, r2],
            params=JoinParams(join_columns=["id"]),
        )

        result = apply_operator(app)

        assert isinstance(result, OperatorSuccess)

    def test_routes_projection(self):
        """Dispatcher routes projection operator correctly."""
        relation = _make_relation(["id", "name", "age"], "employees")
        app = OperatorApplication(
            operator="projection",
            inputs=[relation],
            params=ProjectionParams(columns=["name"]),
        )

        result = apply_operator(app)

        assert isinstance(result, OperatorSuccess)
        assert _get_column_names(result.output) == ["name"]

    def test_routes_cartesian_product(self):
        """Dispatcher routes cartesian product operator correctly."""
        r1 = _make_relation(["a"], "t1")
        r2 = _make_relation(["b"], "t2")
        app = OperatorApplication(
            operator="cartesian_product",
            inputs=[r1, r2],
            params=CartesianProductParams(),
        )

        result = apply_operator(app)

        assert isinstance(result, OperatorSuccess)
        assert _get_column_names(result.output) == ["a", "b"]

    def test_routes_union(self):
        """Dispatcher routes union operator correctly."""
        r1 = _make_relation(["id"], "t1")
        r2 = _make_relation(["id"], "t2")
        app = OperatorApplication(
            operator="union",
            inputs=[r1, r2],
            params=UnionParams(),
        )

        result = apply_operator(app)

        assert isinstance(result, OperatorSuccess)

    def test_routes_difference(self):
        """Dispatcher routes difference operator correctly."""
        r1 = _make_relation(["id"], "t1")
        r2 = _make_relation(["id"], "t2")
        app = OperatorApplication(
            operator="difference",
            inputs=[r1, r2],
            params=DifferenceParams(),
        )

        result = apply_operator(app)

        assert isinstance(result, OperatorSuccess)
        assert _get_column_names(result.output) == ["id"]

    def test_routes_division(self):
        """Dispatcher routes division operator correctly."""
        r1 = _make_relation(["a", "b"], "t1")
        r2 = _make_relation(["b"], "t2")
        app = OperatorApplication(
            operator="division",
            inputs=[r1, r2],
            params=DivisionParams(),
        )

        result = apply_operator(app)

        assert isinstance(result, OperatorSuccess)
        assert _get_column_names(result.output) == ["a"]

    def test_handles_unknown_operator(self):
        """Dispatcher returns failure for unknown operator type."""
        relation = _make_relation(["id"], "t1")
        app = OperatorApplication(
            operator="unknown_op",  # type: ignore
            inputs=[relation],
            params=SelectionParams(),
        )

        result = apply_operator(app)

        assert isinstance(result, OperatorFailure)
        assert "unknown" in result.error.lower() or "Unknown" in result.error

    def test_validates_input_count_unary(self):
        """Dispatcher validates input count for unary operators."""
        r1 = _make_relation(["id"], "t1")
        r2 = _make_relation(["id"], "t2")
        app = OperatorApplication(
            operator="selection",
            inputs=[r1, r2],
            params=SelectionParams(condition=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="id"),
                right=LiteralNode(value=1, data_type="number"),
            )),
        )

        result = apply_operator(app)

        assert isinstance(result, OperatorFailure)
        assert "1" in result.error

    def test_validates_input_count_binary(self):
        """Dispatcher validates input count for binary operators."""
        r1 = _make_relation(["id"], "t1")
        app = OperatorApplication(
            operator="union",
            inputs=[r1],
            params=UnionParams(),
        )

        result = apply_operator(app)

        assert isinstance(result, OperatorFailure)
        assert "2" in result.error



# ---------------------------------------------------------------------------
# Rename operator
# ---------------------------------------------------------------------------

from text_to_sql_planner.operators.rename import apply_rename
from text_to_sql_planner.types.operators import RenameParams


class TestRename:
    """``ρ_{mapping}(R)`` — renames one or more columns of a relation.

    The output's row set equals the input's; only result-variable
    names and free references inside the condition tree change.
    """

    def _table_relation(
        self, table: str, columns: list[str],
    ) -> DRCExpression:
        """Build ``{cols | (cols ∈ table)}``."""
        return DRCExpression(
            result_variables=[ColumnVariable(name=c) for c in columns],
            condition=MembershipNode(variables=list(columns), relation=table),
        )

    def test_simple_rename(self):
        rel = self._table_relation("R", ["a", "b"])
        result = apply_rename(
            RenameParams(mapping={"a": "x"}),
            [rel],
        )
        assert isinstance(result, OperatorSuccess)
        out = result.output
        # Result variables: a → x, b unchanged.
        names = [rv.name for rv in out.result_variables]
        assert names == ["x", "b"]
        # Membership slot: a → x, b unchanged.
        assert isinstance(out.condition, MembershipNode)
        assert out.condition.variables == ["x", "b"]
        assert out.condition.relation == "R"

    def test_rename_multiple_columns(self):
        rel = self._table_relation("R", ["a", "b", "c"])
        result = apply_rename(
            RenameParams(mapping={"a": "x", "c": "z"}),
            [rel],
        )
        assert isinstance(result, OperatorSuccess)
        names = [rv.name for rv in result.output.result_variables]
        assert names == ["x", "b", "z"]

    def test_simultaneous_swap(self):
        """``a ↔ b`` performed as a single simultaneous swap (the
        operator must NOT do ``a → b`` first then ``b → a`` and end
        up with both columns named the same)."""
        rel = self._table_relation("R", ["a", "b"])
        result = apply_rename(
            RenameParams(mapping={"a": "b", "b": "a"}),
            [rel],
        )
        assert isinstance(result, OperatorSuccess)
        out = result.output
        names = [rv.name for rv in out.result_variables]
        # Position 0 (was "a") now "b"; position 1 (was "b") now "a".
        assert names == ["b", "a"]
        # Membership slots reflect the swap.
        assert out.condition.variables == ["b", "a"]

    def test_rename_capture_avoiding(self):
        """Inner quantifier-bound names that happen to coincide with a
        renamed column are not touched. Only free references move."""
        # {a | ∃ a. (a ∈ S) ∧ (a ∈ R)}  -- the outer "a" is the result
        # var, the inner "a" is a quantifier-bound shadow.
        # Construct the inner shadow more carefully so we can tell the
        # rewrite from the leaf:
        outer_membership = MembershipNode(variables=["a"], relation="R")
        inner_existential = QuantifierNode(
            kind="exists",
            variables=["a"],
            body=MembershipNode(variables=["a"], relation="S"),
        )
        condition = LogicalConnectiveNode(
            operator="and",
            left=outer_membership,
            right=inner_existential,
        )
        rel = DRCExpression(
            result_variables=[ColumnVariable(name="a")],
            condition=condition,
        )
        result = apply_rename(
            RenameParams(mapping={"a": "x"}),
            [rel],
        )
        assert isinstance(result, OperatorSuccess)
        out = result.output

        # Result variable renamed.
        assert [rv.name for rv in out.result_variables] == ["x"]

        # Outer membership: free a → x.
        outer = out.condition.left
        assert isinstance(outer, MembershipNode)
        assert outer.variables == ["x"]

        # Inner existential: bound a unchanged (it's shadowed).
        inner = out.condition.right
        assert isinstance(inner, QuantifierNode)
        assert inner.variables == ["a"]
        body = inner.body
        assert isinstance(body, MembershipNode)
        assert body.variables == ["a"]
        assert body.relation == "S"

    def test_rejects_empty_mapping(self):
        rel = self._table_relation("R", ["a"])
        result = apply_rename(RenameParams(mapping={}), [rel])
        assert isinstance(result, OperatorFailure)
        assert "non-empty" in result.error.lower()

    def test_rejects_unknown_column(self):
        rel = self._table_relation("R", ["a", "b"])
        result = apply_rename(
            RenameParams(mapping={"missing": "x"}),
            [rel],
        )
        assert isinstance(result, OperatorFailure)
        assert "missing" in result.error.lower()

    def test_rejects_duplicate_targets(self):
        rel = self._table_relation("R", ["a", "b"])
        result = apply_rename(
            RenameParams(mapping={"a": "x", "b": "x"}),
            [rel],
        )
        assert isinstance(result, OperatorFailure)
        assert "distinct" in result.error.lower()

    def test_rejects_target_collision_with_surviving_column(self):
        """Renaming ``a`` to ``b`` when ``b`` is also a column (not
        being renamed away) creates a name collision."""
        rel = self._table_relation("R", ["a", "b"])
        result = apply_rename(
            RenameParams(mapping={"a": "b"}),
            [rel],
        )
        assert isinstance(result, OperatorFailure)
        assert "collid" in result.error.lower()

    def test_rejects_aggregate_input(self):
        """Aggregate result variables can't be renamed by this operator."""
        from text_to_sql_planner.types.drc import AggregateVariable

        rel = DRCExpression(
            result_variables=[
                ColumnVariable(name="a"),
                AggregateVariable(function="COUNT", column="b"),
            ],
            condition=MembershipNode(variables=["a", "b"], relation="R"),
        )
        result = apply_rename(
            RenameParams(mapping={"a": "x"}),
            [rel],
        )
        assert isinstance(result, OperatorFailure)
        assert "aggregate" in result.error.lower()

    def test_rejects_wrong_input_count(self):
        rel = self._table_relation("R", ["a"])
        result = apply_rename(
            RenameParams(mapping={"a": "x"}),
            [rel, rel],
        )
        assert isinstance(result, OperatorFailure)


class TestRenameDispatcher:
    """The dispatcher routes ``rename`` through ``apply_operator``."""

    def test_dispatcher_routes_rename(self):
        rel = DRCExpression(
            result_variables=[ColumnVariable(name="a")],
            condition=MembershipNode(variables=["a"], relation="R"),
        )
        result = apply_operator(
            OperatorApplication(
                operator="rename",
                inputs=[rel],
                params=RenameParams(mapping={"a": "x"}),
            ),
        )
        assert isinstance(result, OperatorSuccess)
        names = [rv.name for rv in result.output.result_variables]
        assert names == ["x"]

    def test_dispatcher_validates_input_count(self):
        rel = DRCExpression(
            result_variables=[ColumnVariable(name="a")],
            condition=MembershipNode(variables=["a"], relation="R"),
        )
        result = apply_operator(
            OperatorApplication(
                operator="rename",
                inputs=[rel, rel],
                params=RenameParams(mapping={"a": "x"}),
            ),
        )
        assert isinstance(result, OperatorFailure)


# ---------------------------------------------------------------------------
# Aggregate tests (Option 1 fix for dev_585)
# ---------------------------------------------------------------------------

from text_to_sql_planner.operators.aggregate import apply_aggregate
from text_to_sql_planner.types.operators import AggregateParams


class TestAggregate:
    """Tests for the aggregate operator that promotes a column-typed
    result variable to an :class:`AggregateVariable`.

    Without this operator the planner can build the right underlying
    relation (``{x | …}``) but has no way to express the transition
    to the target's aggregate result variable (``{(F x) | …}``).
    These tests pin the structural promotion and the validation
    rules.
    """

    def _single_col_relation(self, name: str = "BountyAmount") -> DRCExpression:
        """Build ``{name | (∃…) name ∈ T}`` — the shape the planner
        produces just before it needs to aggregate."""
        return DRCExpression(
            result_variables=[ColumnVariable(name=name)],
            condition=MembershipNode(
                variables=[name], relation="votes_filtered"
            ),
        )

    def test_promotes_column_to_aggregate(self):
        """Happy path: ``{x | φ}`` becomes ``{(SUM x) | φ}``."""
        from text_to_sql_planner.types.drc import AggregateVariable

        rel = self._single_col_relation("BountyAmount")
        result = apply_aggregate(
            AggregateParams(function="SUM", column="BountyAmount"),
            [rel],
        )
        assert isinstance(result, OperatorSuccess)
        rvs = result.output.result_variables
        assert len(rvs) == 1
        assert isinstance(rvs[0], AggregateVariable)
        assert rvs[0].function == "SUM"
        assert rvs[0].column == "BountyAmount"
        # The condition is preserved verbatim — only the result
        # variable's *kind* changed.
        assert result.output.condition is rel.condition

    @pytest.mark.parametrize(
        "function", ["COUNT", "SUM", "AVG", "MIN", "MAX"]
    )
    def test_accepts_every_supported_aggregate(self, function):
        """All five SQL aggregates round-trip through the operator."""
        from text_to_sql_planner.types.drc import AggregateVariable

        rel = self._single_col_relation("x")
        result = apply_aggregate(
            AggregateParams(function=function, column="x"),
            [rel],
        )
        assert isinstance(result, OperatorSuccess)
        rv = result.output.result_variables[0]
        assert isinstance(rv, AggregateVariable)
        assert rv.function == function

    def test_rejects_unknown_function(self):
        """Functions outside the canonical five are rejected."""
        rel = self._single_col_relation("x")
        # Bypass the type-system by constructing the params manually
        # (the dataclass annotates the field with a Literal but the
        # runtime check still has to fire).
        params = AggregateParams(function="MEDIAN", column="x")  # type: ignore[arg-type]
        result = apply_aggregate(params, [rel])
        assert isinstance(result, OperatorFailure)
        assert "MEDIAN" in result.error

    def test_rejects_zero_inputs(self):
        result = apply_aggregate(
            AggregateParams(function="SUM", column="x"), []
        )
        assert isinstance(result, OperatorFailure)
        assert "1 input" in result.error

    def test_rejects_two_inputs(self):
        rel = self._single_col_relation("x")
        result = apply_aggregate(
            AggregateParams(function="SUM", column="x"), [rel, rel]
        )
        assert isinstance(result, OperatorFailure)
        assert "1 input" in result.error

    def test_rejects_empty_column(self):
        rel = self._single_col_relation("x")
        result = apply_aggregate(
            AggregateParams(function="SUM", column=""), [rel]
        )
        assert isinstance(result, OperatorFailure)

    def test_rejects_input_with_more_than_one_result_variable(self):
        """Aggregate is single-column only — project first if needed."""
        rel = DRCExpression(
            result_variables=[
                ColumnVariable(name="emp_id"),
                ColumnVariable(name="salary"),
            ],
            condition=MembershipNode(
                variables=["emp_id", "salary"], relation="employees"
            ),
        )
        result = apply_aggregate(
            AggregateParams(function="SUM", column="salary"), [rel]
        )
        assert isinstance(result, OperatorFailure)
        assert "exactly one" in result.error.lower()

    def test_rejects_already_aggregate_input(self):
        """Cannot stack aggregate over aggregate."""
        from text_to_sql_planner.types.drc import AggregateVariable

        rel = DRCExpression(
            result_variables=[
                AggregateVariable(function="COUNT", column="x")
            ],
            condition=MembershipNode(variables=["x"], relation="t"),
        )
        result = apply_aggregate(
            AggregateParams(function="SUM", column="x"), [rel]
        )
        assert isinstance(result, OperatorFailure)
        assert "not already an aggregate" in result.error.lower()

    def test_rejects_column_mismatch(self):
        """The ``column`` parameter must match the input's single column."""
        rel = self._single_col_relation("BountyAmount")
        result = apply_aggregate(
            AggregateParams(function="SUM", column="Score"),
            [rel],
        )
        assert isinstance(result, OperatorFailure)
        assert "BountyAmount" in result.error
        assert "Score" in result.error

    def test_dispatcher_routes_to_aggregate(self):
        """``apply_operator`` dispatches the ``"aggregate"`` operator type."""
        from text_to_sql_planner.types.drc import AggregateVariable

        rel = self._single_col_relation("BountyAmount")
        result = apply_operator(
            OperatorApplication(
                operator="aggregate",
                inputs=[rel],
                params=AggregateParams(function="SUM", column="BountyAmount"),
            )
        )
        assert isinstance(result, OperatorSuccess)
        assert isinstance(
            result.output.result_variables[0], AggregateVariable
        )

    def test_dispatcher_validates_unary_arity(self):
        """The dispatcher's unary-input check applies to aggregate too."""
        rel = self._single_col_relation()
        result = apply_operator(
            OperatorApplication(
                operator="aggregate",
                inputs=[rel, rel],
                params=AggregateParams(function="SUM", column="BountyAmount"),
            )
        )
        assert isinstance(result, OperatorFailure)


class TestAggregateUnblocksDev585Pattern:
    """End-to-end pin for the bug behind dev_585.

    Before the fix: the planner builds ``{BountyAmount | …}`` and
    then has no operator that can transition to the target
    ``{(SUM BountyAmount) | …}``. ``rename`` can only relabel the
    column name; ``projection`` can introduce an aggregate via its
    column-spec parser but only over an existing column with that
    underlying name (which is the same one already in the relation,
    so it produces a structurally-identical DRC and gets
    de-duplicated).

    After the fix: the planner can apply ``aggregate`` as a single
    operator step.
    """

    def test_reproduces_the_required_structural_promotion(self):
        from text_to_sql_planner.types.drc import AggregateVariable

        # The "votes joined with posts-about-data, projected to
        # BountyAmount" relation, simplified for the test.
        underlying = DRCExpression(
            result_variables=[ColumnVariable(name="BountyAmount")],
            condition=MembershipNode(
                variables=["BountyAmount"], relation="votes_filtered"
            ),
        )

        # The target shape the question converter produced.
        target_rv = AggregateVariable(function="SUM", column="BountyAmount")

        promoted = apply_aggregate(
            AggregateParams(function="SUM", column="BountyAmount"),
            [underlying],
        )

        assert isinstance(promoted, OperatorSuccess)
        out_rv = promoted.output.result_variables[0]
        assert isinstance(out_rv, AggregateVariable)
        assert out_rv.function == target_rv.function
        assert out_rv.column == target_rv.column
        # And the underlying tuple-binding is unchanged.
        assert promoted.output.condition is underlying.condition
