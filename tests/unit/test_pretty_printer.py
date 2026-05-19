"""Unit tests for the pretty printer (Unicode notation)."""

import pytest

from text_to_sql_planner.printer import pretty_print, PrintSuccess, PrintFailure
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


class TestSimpleExpressions:
    """Test simple expressions with variable refs and literals."""

    def test_simple_variable_ref(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | x}"

    def test_multiple_result_variables(self):
        expr = DRCExpression(
            result_variables=[
                ColumnVariable(name="name"),
                ColumnVariable(name="age"),
            ],
            condition=VariableRefNode(name="name"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{name, age | name}"

    def test_string_literal(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LiteralNode(value="hello", data_type="string"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == '{x | "hello"}'

    def test_number_literal(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LiteralNode(value=42, data_type="number"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | 42}"

    def test_comparison_expression(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="x"),
                right=LiteralNode(value="hello", data_type="string"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == '{x | x = "hello"}'

    def test_comparison_greater_than(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ComparisonNode(
                operator=">",
                left=VariableRefNode(name="score"),
                right=LiteralNode(value=90, data_type="number"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | score > 90}"


class TestQuantifiers:
    """Test quantifier formatting with Unicode symbols."""

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
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | \u2203 y,z \u2208 Enrolled (y)}"

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
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | \u2200 a \u2208 Grades (a > 90)}"

    def test_exists_with_multiple_variables(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="name")],
            condition=QuantifierNode(
                kind="exists",
                variables=["sid", "cid", "grade"],
                relation="Enrollment",
                body=ComparisonNode(
                    operator="=",
                    left=VariableRefNode(name="cid"),
                    right=LiteralNode(value="CS101", data_type="string"),
                ),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == '{name | \u2203 sid,cid,grade \u2208 Enrollment (cid = "CS101")}'


class TestLogicalConnectives:
    """Test logical connectives with precedence parenthesization."""

    def test_and_connective(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="and",
                left=VariableRefNode(name="a"),
                right=VariableRefNode(name="b"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | a \u2227 b}"

    def test_or_connective(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="or",
                left=VariableRefNode(name="a"),
                right=VariableRefNode(name="b"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | a \u2228 b}"

    def test_implies_connective(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="implies",
                left=VariableRefNode(name="p"),
                right=VariableRefNode(name="q"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | p \u2192 q}"

    def test_not_expression(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=NotNode(operand=VariableRefNode(name="x")),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | \u00ACx}"

    def test_and_inside_or_needs_no_parens(self):
        """∧ binds tighter than ∨, so a ∧ b inside ∨ doesn't need parens."""
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="or",
                left=LogicalConnectiveNode(
                    operator="and",
                    left=VariableRefNode(name="a"),
                    right=VariableRefNode(name="b"),
                ),
                right=VariableRefNode(name="c"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        # ∧ has higher precedence than ∨, no parens needed
        assert result.output == "{x | a \u2227 b \u2228 c}"

    def test_or_inside_and_needs_parens(self):
        """∨ binds looser than ∧, so a ∨ b inside ∧ needs parens."""
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="and",
                left=LogicalConnectiveNode(
                    operator="or",
                    left=VariableRefNode(name="a"),
                    right=VariableRefNode(name="b"),
                ),
                right=VariableRefNode(name="c"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        # ∨ has lower precedence than ∧, needs parens
        assert result.output == "{x | (a \u2228 b) \u2227 c}"

    def test_implies_inside_and_needs_parens(self):
        """→ has lowest precedence, needs parens inside ∧."""
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="and",
                left=LogicalConnectiveNode(
                    operator="implies",
                    left=VariableRefNode(name="p"),
                    right=VariableRefNode(name="q"),
                ),
                right=VariableRefNode(name="r"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | (p \u2192 q) \u2227 r}"

    def test_not_with_and(self):
        """¬ has highest precedence, no parens needed around it."""
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="and",
                left=NotNode(operand=VariableRefNode(name="a")),
                right=VariableRefNode(name="b"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | \u00ACa \u2227 b}"

    def test_nested_and_same_precedence_no_parens(self):
        """Same-precedence operators don't need extra parens."""
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="and",
                left=LogicalConnectiveNode(
                    operator="and",
                    left=VariableRefNode(name="a"),
                    right=VariableRefNode(name="b"),
                ),
                right=VariableRefNode(name="c"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | a \u2227 b \u2227 c}"


class TestMembership:
    """Test membership formatting."""

    def test_simple_membership(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=MembershipNode(variables=["x", "y"], relation="Students"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | x,y \u2208 Students}"

    def test_single_variable_membership(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="id")],
            condition=MembershipNode(variables=["id"], relation="Courses"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{id | id \u2208 Courses}"


class TestAggregates:
    """Test aggregate result variables."""

    def test_count_aggregate(self):
        expr = DRCExpression(
            result_variables=[AggregateVariable(function="COUNT", column="id")],
            condition=VariableRefNode(name="id"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{COUNT(id) | id}"

    def test_sum_aggregate(self):
        expr = DRCExpression(
            result_variables=[AggregateVariable(function="SUM", column="amount")],
            condition=VariableRefNode(name="amount"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{SUM(amount) | amount}"

    def test_mixed_result_variables(self):
        expr = DRCExpression(
            result_variables=[
                ColumnVariable(name="name"),
                AggregateVariable(function="AVG", column="grade"),
            ],
            condition=VariableRefNode(name="name"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{name, AVG(grade) | name}"

    def test_multiple_aggregates(self):
        expr = DRCExpression(
            result_variables=[
                AggregateVariable(function="MIN", column="price"),
                AggregateVariable(function="MAX", column="price"),
            ],
            condition=VariableRefNode(name="price"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{MIN(price), MAX(price) | price}"


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
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | a + 1}"

    def test_multiplication(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ArithmeticNode(
                operator="*",
                left=VariableRefNode(name="price"),
                right=VariableRefNode(name="qty"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | price * qty}"

    def test_addition_inside_multiplication_needs_parens(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ArithmeticNode(
                operator="*",
                left=ArithmeticNode(
                    operator="+",
                    left=VariableRefNode(name="a"),
                    right=VariableRefNode(name="b"),
                ),
                right=VariableRefNode(name="c"),
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | (a + b) * c}"

    def test_multiplication_inside_addition_no_parens(self):
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
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        assert result.output == "{x | a * b + c}"


class TestComplexExpressions:
    """Test full complex DRC expressions."""

    def test_complex_query(self):
        """Find students enrolled in CS101 with grade > 80."""
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="student_name")],
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
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        expected = (
            '{student_name | '
            '\u2203 sid,cid,grade \u2208 Enrollment (cid = "CS101" \u2227 grade > 80) '
            '\u2227 student_name,sid \u2208 Students}'
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
        result = pretty_print(expr)
        assert isinstance(result, PrintSuccess)
        expected = "{x | \u2200 s \u2208 Students (s \u2208 Enrolled \u2192 grade > 0)}"
        assert result.output == expected


class TestErrorCases:
    """Test error handling for malformed inputs."""

    def test_none_input(self):
        result = pretty_print(None)
        assert isinstance(result, PrintFailure)
        assert "None" in result.error.message

    def test_non_drc_expression_input(self):
        result = pretty_print("not an expression")
        assert isinstance(result, PrintFailure)
        assert "Expected DRCExpression" in result.error.message

    def test_none_condition(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=None,
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintFailure)
        assert "None" in result.error.message

    def test_empty_variable_name(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name=""),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintFailure)
        assert "empty name" in result.error.message

    def test_none_result_variable(self):
        expr = DRCExpression(
            result_variables=[None],
            condition=VariableRefNode(name="x"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintFailure)
        assert "None" in result.error.message

    def test_empty_column_variable_name(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="")],
            condition=VariableRefNode(name="x"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintFailure)
        assert "empty name" in result.error.message

    def test_aggregate_empty_function(self):
        expr = DRCExpression(
            result_variables=[AggregateVariable(function="", column="x")],
            condition=VariableRefNode(name="x"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintFailure)
        assert "empty function" in result.error.message

    def test_aggregate_empty_column(self):
        expr = DRCExpression(
            result_variables=[AggregateVariable(function="COUNT", column="")],
            condition=VariableRefNode(name="x"),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintFailure)
        assert "empty column" in result.error.message

    def test_membership_empty_relation(self):
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=MembershipNode(variables=["x"], relation=""),
        )
        result = pretty_print(expr)
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
        result = pretty_print(expr)
        assert isinstance(result, PrintFailure)
        assert "None" in result.error.message

    def test_all_or_nothing_no_partial_output(self):
        """Ensure that on error, no partial string is produced."""
        expr = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=LogicalConnectiveNode(
                operator="and",
                left=VariableRefNode(name="valid"),
                right=None,
            ),
        )
        result = pretty_print(expr)
        assert isinstance(result, PrintFailure)
        assert not hasattr(result, "output") or not isinstance(result, PrintSuccess)

    def test_integer_input(self):
        result = pretty_print(123)
        assert isinstance(result, PrintFailure)
        assert "Expected DRCExpression" in result.error.message
