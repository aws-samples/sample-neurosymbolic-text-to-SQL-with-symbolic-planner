"""Unit tests for the recursive descent DRC parser."""

import pytest

from text_to_sql_planner.parser import parse, ParserSuccess, ParserFailure
from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ArithmeticNode,
    ColumnVariable,
    ComparisonNode,
    DRCExpression,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)


class TestVariableReference:
    """Test parsing of bare symbol variable references."""

    def test_simple_variable_in_condition(self):
        result = parse("(drc (x) x)")
        assert isinstance(result, ParserSuccess)
        expr = result.expression
        assert len(expr.result_variables) == 1
        assert isinstance(expr.result_variables[0], ColumnVariable)
        assert expr.result_variables[0].name == "x"
        assert isinstance(expr.condition, VariableRefNode)
        assert expr.condition.name == "x"

    def test_multiple_result_variables(self):
        result = parse("(drc (x y z) x)")
        assert isinstance(result, ParserSuccess)
        expr = result.expression
        assert len(expr.result_variables) == 3
        assert expr.result_variables[0].name == "x"
        assert expr.result_variables[1].name == "y"
        assert expr.result_variables[2].name == "z"


class TestMembership:
    """Test parsing of membership expressions."""

    def test_simple_membership(self):
        result = parse("(drc (x) (in (x y) Employees))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, MembershipNode)
        assert cond.variables == ["x", "y"]
        assert cond.relation == "Employees"

    def test_single_variable_membership(self):
        result = parse("(drc (id) (in (id) Users))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, MembershipNode)
        assert cond.variables == ["id"]
        assert cond.relation == "Users"


class TestQuantifiers:
    """Test parsing of forall and exists quantifiers."""

    def test_exists_quantifier(self):
        result = parse("(drc (x) (exists (y) (and (in (y) Orders) (= x y))))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, QuantifierNode)
        assert cond.kind == "exists"
        assert cond.variables == ["y"]
        assert isinstance(cond.body, LogicalConnectiveNode)

    def test_forall_quantifier(self):
        result = parse("(drc (x) (forall (y z) (in (y z) Products)))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, QuantifierNode)
        assert cond.kind == "forall"
        assert cond.variables == ["y", "z"]
        assert isinstance(cond.body, MembershipNode)

    def test_nested_quantifiers(self):
        result = parse("(drc (x) (exists (y) (forall (z) (= y z))))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, QuantifierNode)
        assert cond.kind == "exists"
        assert isinstance(cond.body, QuantifierNode)
        assert cond.body.kind == "forall"


class TestLogicalConnectives:
    """Test parsing of logical connectives."""

    def test_and(self):
        result = parse("(drc (x) (and x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, LogicalConnectiveNode)
        assert cond.operator == "and"
        assert isinstance(cond.left, VariableRefNode)
        assert cond.left.name == "x"
        assert isinstance(cond.right, VariableRefNode)
        assert cond.right.name == "y"

    def test_or(self):
        result = parse("(drc (x) (or x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, LogicalConnectiveNode)
        assert cond.operator == "or"

    def test_implies(self):
        result = parse("(drc (x) (implies x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, LogicalConnectiveNode)
        assert cond.operator == "implies"

    def test_not(self):
        result = parse("(drc (x) (not x))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, NotNode)
        assert isinstance(cond.operand, VariableRefNode)
        assert cond.operand.name == "x"

    def test_nested_logical(self):
        result = parse("(drc (x) (and (not x) (or y z)))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, LogicalConnectiveNode)
        assert cond.operator == "and"
        assert isinstance(cond.left, NotNode)
        assert isinstance(cond.right, LogicalConnectiveNode)
        assert cond.right.operator == "or"


class TestComparisons:
    """Test parsing of comparison operators."""

    def test_equals(self):
        result = parse("(drc (x) (= x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert cond.operator == "="
        assert isinstance(cond.left, VariableRefNode)
        assert isinstance(cond.right, VariableRefNode)

    def test_not_equals(self):
        result = parse("(drc (x) (!= x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert cond.operator == "!="

    def test_less_than(self):
        result = parse("(drc (x) (< x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert cond.operator == "<"

    def test_greater_than(self):
        result = parse("(drc (x) (> x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert cond.operator == ">"

    def test_less_than_or_equal(self):
        result = parse("(drc (x) (<= x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert cond.operator == "<="

    def test_greater_than_or_equal(self):
        result = parse("(drc (x) (>= x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert cond.operator == ">="

    def test_comparison_with_literal(self):
        result = parse('(drc (x) (= x 42))')
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert isinstance(cond.left, VariableRefNode)
        assert isinstance(cond.right, LiteralNode)
        assert cond.right.value == 42
        assert cond.right.data_type == "number"


class TestArithmetic:
    """Test parsing of arithmetic operators."""

    def test_addition(self):
        result = parse("(drc (x) (+ x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ArithmeticNode)
        assert cond.operator == "+"

    def test_subtraction(self):
        result = parse("(drc (x) (- x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ArithmeticNode)
        assert cond.operator == "-"

    def test_multiplication(self):
        result = parse("(drc (x) (* x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ArithmeticNode)
        assert cond.operator == "*"

    def test_division(self):
        result = parse("(drc (x) (/ x y))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ArithmeticNode)
        assert cond.operator == "/"

    def test_nested_arithmetic(self):
        result = parse("(drc (x) (+ (* x y) (- a b)))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ArithmeticNode)
        assert cond.operator == "+"
        assert isinstance(cond.left, ArithmeticNode)
        assert cond.left.operator == "*"
        assert isinstance(cond.right, ArithmeticNode)
        assert cond.right.operator == "-"


class TestAggregates:
    """Test parsing of aggregate functions in result variables."""

    def test_count_aggregate(self):
        result = parse("(drc ((COUNT id)) x)")
        assert isinstance(result, ParserSuccess)
        rv = result.expression.result_variables[0]
        assert isinstance(rv, AggregateVariable)
        assert rv.function == "COUNT"
        assert rv.column == "id"

    def test_sum_aggregate(self):
        result = parse("(drc ((SUM amount)) x)")
        assert isinstance(result, ParserSuccess)
        rv = result.expression.result_variables[0]
        assert isinstance(rv, AggregateVariable)
        assert rv.function == "SUM"
        assert rv.column == "amount"

    def test_avg_aggregate(self):
        result = parse("(drc ((AVG score)) x)")
        assert isinstance(result, ParserSuccess)
        rv = result.expression.result_variables[0]
        assert isinstance(rv, AggregateVariable)
        assert rv.function == "AVG"
        assert rv.column == "score"

    def test_min_aggregate(self):
        result = parse("(drc ((MIN price)) x)")
        assert isinstance(result, ParserSuccess)
        rv = result.expression.result_variables[0]
        assert isinstance(rv, AggregateVariable)
        assert rv.function == "MIN"

    def test_max_aggregate(self):
        result = parse("(drc ((MAX price)) x)")
        assert isinstance(result, ParserSuccess)
        rv = result.expression.result_variables[0]
        assert isinstance(rv, AggregateVariable)
        assert rv.function == "MAX"

    def test_mixed_result_variables(self):
        result = parse("(drc (name (COUNT id) (SUM amount)) x)")
        assert isinstance(result, ParserSuccess)
        rvs = result.expression.result_variables
        assert len(rvs) == 3
        assert isinstance(rvs[0], ColumnVariable)
        assert rvs[0].name == "name"
        assert isinstance(rvs[1], AggregateVariable)
        assert rvs[1].function == "COUNT"
        assert isinstance(rvs[2], AggregateVariable)
        assert rvs[2].function == "SUM"


class TestFullDRCExpression:
    """Test parsing of complete DRC expressions."""

    def test_full_expression(self):
        source = "(drc (name salary) (and (in (name salary) Employees) (> salary 50000)))"
        result = parse(source)
        assert isinstance(result, ParserSuccess)
        expr = result.expression
        assert len(expr.result_variables) == 2
        assert expr.result_variables[0].name == "name"
        assert expr.result_variables[1].name == "salary"
        cond = expr.condition
        assert isinstance(cond, LogicalConnectiveNode)
        assert cond.operator == "and"
        assert isinstance(cond.left, MembershipNode)
        assert isinstance(cond.right, ComparisonNode)

    def test_complex_expression_with_quantifier(self):
        source = "(drc (x) (exists (y) (and (in (x y) Orders) (= x 1))))"
        result = parse(source)
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, QuantifierNode)
        assert cond.kind == "exists"
        assert isinstance(cond.body, LogicalConnectiveNode)


class TestLiterals:
    """Test parsing of string and number literals."""

    def test_string_literal(self):
        result = parse('(drc (x) (= x "hello"))')
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert isinstance(cond.right, LiteralNode)
        assert cond.right.value == "hello"
        assert cond.right.data_type == "string"

    def test_integer_literal(self):
        result = parse("(drc (x) (= x 42))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert isinstance(cond.right, LiteralNode)
        assert cond.right.value == 42
        assert cond.right.data_type == "number"

    def test_float_literal(self):
        result = parse("(drc (x) (= x 3.14))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert isinstance(cond.right, LiteralNode)
        assert cond.right.value == 3.14
        assert cond.right.data_type == "number"

    def test_negative_number_literal(self):
        result = parse("(drc (x) (= x -5))")
        assert isinstance(result, ParserSuccess)
        cond = result.expression.condition
        assert isinstance(cond, ComparisonNode)
        assert isinstance(cond.right, LiteralNode)
        assert cond.right.value == -5
        assert cond.right.data_type == "number"


class TestErrorCases:
    """Test error handling and failure cases."""

    def test_unmatched_open_paren(self):
        result = parse("(drc (x) (= x y)")
        assert isinstance(result, ParserFailure)
        assert result.error.offset >= 0
        assert result.error.message

    def test_unmatched_close_paren(self):
        result = parse("(drc (x) x))")
        assert isinstance(result, ParserFailure)
        assert result.error.offset >= 0

    def test_unknown_operator(self):
        result = parse("(drc (x) (foobar x y))")
        assert isinstance(result, ParserFailure)
        assert "Unknown operator" in result.error.message

    def test_nesting_depth_exceeded(self):
        # Build a deeply nested expression > 50 levels
        # Each level of nesting adds a (not ...) wrapper
        inner = "x"
        for _ in range(52):
            inner = f"(not {inner})"
        source = f"(drc (x) {inner})"
        result = parse(source)
        assert isinstance(result, ParserFailure)
        assert "max nesting depth exceeded" in result.error.message

    def test_empty_input(self):
        result = parse("")
        assert isinstance(result, ParserFailure)

    def test_missing_drc_keyword(self):
        result = parse("(foo (x) x)")
        assert isinstance(result, ParserFailure)
        assert "drc" in result.error.message

    def test_missing_result_variables(self):
        result = parse("(drc x)")
        assert isinstance(result, ParserFailure)

    def test_unknown_aggregate_function(self):
        result = parse("(drc ((MEDIAN x)) y)")
        assert isinstance(result, ParserFailure)
        assert "Unknown aggregate function" in result.error.message
