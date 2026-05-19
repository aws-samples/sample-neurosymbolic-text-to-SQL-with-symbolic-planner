"""Unit tests for the Lisp S-expression printer."""

import pytest

from text_to_sql_planner.printer import print_lisp, PrintSuccess, PrintFailure
from text_to_sql_planner.types.drc import (
    DRCExpression,
    ColumnVariable,
    AggregateVariable,
    QuantifierNode,
    LogicalConnectiveNode,
    NotNode,
    ComparisonNode,
    MembershipNode,
    ArithmeticNode,
    LiteralNode,
    VariableRefNode,
)


class TestVariableReference:
    """Test simple variable reference expressions."""

    def test_simple_variable_ref(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) x)"

    def test_multiple_result_variables(self):
        expr = DRCExpression(
            result_variables=[
                ColumnVariable(name="name"),
                ColumnVariable(name="age"),
            ],
            condition=VariableRefNode(name="name"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (name age) name)"


class TestMembership:
    """Test membership expressions."""

    def test_simple_membership(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=MembershipNode(variables=["x", "y"], relation="Students"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (in (x y) Students))"

    def test_single_variable_membership(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="id")],
            condition=MembershipNode(variables=["id"], relation="Courses"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (id) (in (id) Courses))"


class TestQuantifiers:
    """Test quantifier expressions."""

    def test_exists_quantifier(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=QuantifierNode(
                kind="exists",
                variables=["y", "z"],
                relation="Enrolled",
                body=VariableRefNode(name="y"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (exists (y z) Enrolled y))"

    def test_forall_quantifier(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=QuantifierNode(
                kind="forall",
                variables=["a"],
                relation="Grades",
                body=ComparisonNode(
                    operator=">",
                    left=VariableRefNode(name="a"),
                    right=LiteralNode(value=90, data_type="number"),
                ),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (forall (a) Grades (> a 90)))"


class TestLogicalConnectives:
    """Test logical connective expressions."""

    def test_and_connective(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="and",
                left=VariableRefNode(name="x"),
                right=VariableRefNode(name="y"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (and x y))"

    def test_or_connective(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="or",
                left=VariableRefNode(name="a"),
                right=VariableRefNode(name="b"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (or a b))"

    def test_implies_connective(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="implies",
                left=VariableRefNode(name="p"),
                right=VariableRefNode(name="q"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (implies p q))"

    def test_not_expression(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=NotNode(operand=VariableRefNode(name="x")),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (not x))"


class TestComparisons:
    """Test comparison expressions."""

    def test_equality(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="x"),
                right=LiteralNode(value="hello", data_type="string"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == '(drc (x) (= x "hello"))'

    def test_not_equal(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ComparisonNode(
                operator="!=",
                left=VariableRefNode(name="x"),
                right=LiteralNode(value=0, data_type="number"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (!= x 0))"

    def test_less_than(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ComparisonNode(
                operator="<",
                left=VariableRefNode(name="a"),
                right=VariableRefNode(name="b"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (< a b))"

    def test_greater_equal(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ComparisonNode(
                operator=">=",
                left=VariableRefNode(name="score"),
                right=LiteralNode(value=50, data_type="number"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (>= score 50))"


class TestArithmetic:
    """Test arithmetic expressions."""

    def test_addition(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ArithmeticNode(
                operator="+",
                left=VariableRefNode(name="a"),
                right=LiteralNode(value=1, data_type="number"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (+ a 1))"

    def test_multiplication(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ArithmeticNode(
                operator="*",
                left=VariableRefNode(name="price"),
                right=VariableRefNode(name="quantity"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (* price quantity))"

    def test_nested_arithmetic(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ArithmeticNode(
                operator="+",
                left=ArithmeticNode(
                    operator="*",
                    left=VariableRefNode(name="a"),
                    right=VariableRefNode(name="b"),
                ),
                right=VariableRefNode(name="c"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (x) (+ (* a b) c))"


class TestAggregates:
    """Test aggregate result variables."""

    def test_count_aggregate(self):
        expr = DRCExpression(
            result_variables=[AggregateVariable(function="COUNT", column="student_id")],
            condition=VariableRefNode(name="student_id"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc ((COUNT student_id)) student_id)"

    def test_sum_aggregate(self):
        expr = DRCExpression(
            result_variables=[AggregateVariable(function="SUM", column="amount")],
            condition=VariableRefNode(name="amount"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc ((SUM amount)) amount)"

    def test_mixed_result_variables(self):
        expr = DRCExpression(
            result_variables=[
                ColumnVariable(name="name"),
                AggregateVariable(function="AVG", column="grade"),
            ],
            condition=VariableRefNode(name="name"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "(drc (name (AVG grade)) name)"


class TestFullDRCExpression:
    """Test full DRC expression round-trip formatting."""

    def test_complex_drc_expression(self):
        """A realistic DRC query: find students enrolled in CS101 with grade > 80."""
        expr = DRCExpression(
            result_variables=[
                ColumnVariable(name="student_name"),
            ],
            condition=LogicalConnectiveNode(
                operator="and",
                left=QuantifierNode(
                    kind="exists",
                    variables=["sid", "cid", "grade"],
                    relation="Enrollment",
                    body=LogicalConnectiveNode(
                        operator="and",
                        left=ComparisonNode(
                            operator="=",
                            left=VariableRefNode(name="cid"),
                            right=LiteralNode(value="CS101", data_type="string"),
                        ),
                        right=ComparisonNode(
                            operator=">",
                            left=VariableRefNode(name="grade"),
                            right=LiteralNode(value=80, data_type="number"),
                        ),
                    ),
                ),
                right=MembershipNode(variables=["student_name", "sid"], relation="Students"),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        expected = (
            '(drc (student_name) (and '
            '(exists (sid cid grade) Enrollment (and (= cid "CS101") (> grade 80))) '
            '(in (student_name sid) Students)))'
        )
        assert result.output == expected

    def test_forall_with_implies(self):
        """For all students, if enrolled then grade > 0."""
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=QuantifierNode(
                kind="forall",
                variables=["s"],
                relation="Students",
                body=LogicalConnectiveNode(
                    operator="implies",
                    left=MembershipNode(variables=["s"], relation="Enrolled"),
                    right=ComparisonNode(
                        operator=">",
                        left=VariableRefNode(name="grade"),
                        right=LiteralNode(value=0, data_type="number"),
                    ),
                ),
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintSuccess)
        expected = "(drc (x) (forall (s) Students (implies (in (s) Enrolled) (> grade 0))))"
        assert result.output == expected


class TestErrorCases:
    """Test error handling for malformed inputs."""

    def test_none_input(self):
        result = print_lisp(None)
        assert isinstance(result, PrintFailure)
        assert "None" in result.error.message

    def test_non_drc_expression_input(self):
        result = print_lisp("not an expression")
        assert isinstance(result, PrintFailure)
        assert "Expected DRCExpression" in result.error.message

    def test_none_condition(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=None,
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintFailure)
        assert "None" in result.error.message

    def test_empty_variable_name(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name=""),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintFailure)
        assert "empty name" in result.error.message

    def test_none_result_variable(self):
        expr = DRCExpression(
            result_variables=[None],
            condition=VariableRefNode(name="x"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintFailure)
        assert "None" in result.error.message

    def test_empty_column_variable_name(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="")],
            condition=VariableRefNode(name="x"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintFailure)
        assert "empty name" in result.error.message

    def test_aggregate_empty_function(self):
        expr = DRCExpression(
            result_variables=[AggregateVariable(function="", column="x")],
            condition=VariableRefNode(name="x"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintFailure)
        assert "empty function" in result.error.message

    def test_aggregate_empty_column(self):
        expr = DRCExpression(
            result_variables=[AggregateVariable(function="COUNT", column="")],
            condition=VariableRefNode(name="x"),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintFailure)
        assert "empty column" in result.error.message

    def test_membership_empty_relation(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=MembershipNode(variables=["x"], relation=""),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintFailure)
        assert "empty relation" in result.error.message

    def test_none_body_in_quantifier(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=QuantifierNode(
                kind="exists",
                variables=["y"],
                relation="R",
                body=None,
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintFailure)
        assert "None" in result.error.message

    def test_all_or_nothing_no_partial_output(self):
        """Ensure that on error, no partial string is produced."""
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="and",
                left=VariableRefNode(name="valid"),
                right=None,  # This will cause an error
            ),
        )
        result = print_lisp(expr)
        assert isinstance(result, PrintFailure)
        # No output attribute on failure
        assert not hasattr(result, "output") or not isinstance(result, PrintSuccess)
