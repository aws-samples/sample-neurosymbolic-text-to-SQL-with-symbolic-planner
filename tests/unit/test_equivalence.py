"""Unit tests for SMT-LIB converter and equivalence checker."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from text_to_sql_planner.equivalence import (
    EquivalenceCheckerConfig,
    EquivalentResult,
    IndeterminateResult,
    NotEquivalentResult,
    check_equivalence,
    convert_to_smt,
)
from text_to_sql_planner.types.drc import (
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


# ============================================================
# SMT-LIB Converter Tests
# ============================================================


class TestConvertToSmt:
    """Tests for convert_to_smt function."""

    def test_simple_variable_ref(self):
        """A variable reference should produce a declare-const and assert."""
        node = VariableRefNode(name="x")
        result = convert_to_smt(node)

        assert "(set-logic ALL)" in result
        assert "(declare-const x Int)" in result
        assert "(assert x)" in result
        assert "(check-sat)" in result

    def test_literal_number(self):
        """Numeric literals should be rendered as numerals."""
        node = LiteralNode(value=42, data_type="number")
        result = convert_to_smt(node)

        assert "(assert 42)" in result

    def test_literal_string(self):
        """String literals should be rendered as SMT-LIB string literals."""
        node = LiteralNode(value="hello", data_type="string")
        result = convert_to_smt(node)

        assert '(assert "hello")' in result

    def test_comparison_equals(self):
        """Equality comparison should use (= ...)."""
        node = ComparisonNode(
            operator="=",
            left=VariableRefNode(name="x"),
            right=LiteralNode(value=5, data_type="number"),
        )
        result = convert_to_smt(node)

        assert "(= x 5)" in result
        assert "(declare-const x Int)" in result

    def test_comparison_not_equals(self):
        """Not-equals should use (distinct ...)."""
        node = ComparisonNode(
            operator="!=",
            left=VariableRefNode(name="a"),
            right=VariableRefNode(name="b"),
        )
        result = convert_to_smt(node)

        assert "(distinct a b)" in result

    def test_comparison_less_than(self):
        """Less-than should use (< ...)."""
        node = ComparisonNode(
            operator="<",
            left=VariableRefNode(name="x"),
            right=LiteralNode(value=10, data_type="number"),
        )
        result = convert_to_smt(node)

        assert "(< x 10)" in result

    def test_comparison_greater_equal(self):
        """Greater-or-equal should use (>= ...)."""
        node = ComparisonNode(
            operator=">=",
            left=VariableRefNode(name="y"),
            right=LiteralNode(value=0, data_type="number"),
        )
        result = convert_to_smt(node)

        assert "(>= y 0)" in result

    def test_logical_and(self):
        """Logical AND should use (and ...)."""
        node = LogicalConnectiveNode(
            operator="and",
            left=VariableRefNode(name="p"),
            right=VariableRefNode(name="q"),
        )
        result = convert_to_smt(node)

        assert "(and p q)" in result

    def test_logical_or(self):
        """Logical OR should use (or ...)."""
        node = LogicalConnectiveNode(
            operator="or",
            left=VariableRefNode(name="a"),
            right=VariableRefNode(name="b"),
        )
        result = convert_to_smt(node)

        assert "(or a b)" in result

    def test_logical_implies(self):
        """Implication should use (=> ...)."""
        node = LogicalConnectiveNode(
            operator="implies",
            left=VariableRefNode(name="p"),
            right=VariableRefNode(name="q"),
        )
        result = convert_to_smt(node)

        assert "(=> p q)" in result

    def test_not(self):
        """Negation should use (not ...)."""
        node = NotNode(operand=VariableRefNode(name="x"))
        result = convert_to_smt(node)

        assert "(not x)" in result

    def test_membership(self):
        """Membership should declare relation as uninterpreted function."""
        node = MembershipNode(
            variables=["x", "y"],
            relation="employees",
        )
        result = convert_to_smt(node)

        assert "(declare-fun employees (Int Int) Bool)" in result
        assert "(employees x y)" in result

    def test_quantifier_forall(self):
        """Forall quantifier should use (forall ((var Int)) body)."""
        node = QuantifierNode(
            kind="forall",
            variables=["x"],
            body=VariableRefNode(name="x"),
        )
        result = convert_to_smt(node)

        assert "forall" in result
        assert "(x Int)" in result

    def test_quantifier_exists(self):
        """Exists quantifier should use (exists ((var Int)) body)."""
        node = QuantifierNode(
            kind="exists",
            variables=["y"],
            body=ComparisonNode(
                operator=">",
                left=VariableRefNode(name="y"),
                right=LiteralNode(value=0, data_type="number"),
            ),
        )
        result = convert_to_smt(node)

        assert "exists" in result
        assert "(y Int)" in result
        assert "(> y 0)" in result

    def test_quantifier_multiple_variables(self):
        """Multiple quantified variables should all appear in bindings."""
        node = QuantifierNode(
            kind="exists",
            variables=["a", "b", "c"],
            body=MembershipNode(variables=["a", "b", "c"], relation="T"),
        )
        result = convert_to_smt(node)

        assert "(a Int)" in result
        assert "(b Int)" in result
        assert "(c Int)" in result
        assert "(declare-fun T (Int Int Int) Bool)" in result

    def test_arithmetic_addition(self):
        """Addition should use (+ ...)."""
        node = ArithmeticNode(
            operator="+",
            left=VariableRefNode(name="x"),
            right=LiteralNode(value=1, data_type="number"),
        )
        result = convert_to_smt(node)

        assert "(+ x 1)" in result

    def test_arithmetic_multiplication(self):
        """Multiplication should use (* ...)."""
        node = ArithmeticNode(
            operator="*",
            left=VariableRefNode(name="a"),
            right=VariableRefNode(name="b"),
        )
        result = convert_to_smt(node)

        assert "(* a b)" in result

    def test_nested_expression(self):
        """Complex nested expressions should convert correctly."""
        # (x > 0) AND (membership in R)
        node = LogicalConnectiveNode(
            operator="and",
            left=ComparisonNode(
                operator=">",
                left=VariableRefNode(name="x"),
                right=LiteralNode(value=0, data_type="number"),
            ),
            right=MembershipNode(variables=["x", "y"], relation="R"),
        )
        result = convert_to_smt(node)

        assert "(and (> x 0) (R x y))" in result
        assert "(declare-fun R (Int Int) Bool)" in result
        assert "(declare-const x Int)" in result
        assert "(declare-const y Int)" in result

    def test_complete_script_structure(self):
        """The output should be a valid SMT-LIB script structure."""
        node = ComparisonNode(
            operator="=",
            left=VariableRefNode(name="x"),
            right=LiteralNode(value=1, data_type="number"),
        )
        result = convert_to_smt(node)
        lines = result.split("\n")

        # First line should be set-logic
        assert lines[0] == "(set-logic ALL)"
        # Last line should be check-sat
        assert lines[-1] == "(check-sat)"
        # Should have an assert line
        assert any(line.startswith("(assert") for line in lines)


# ============================================================
# Equivalence Checker Tests
# ============================================================


class TestEquivalenceCheckerArityMismatch:
    """Test early exit when result variable counts differ."""

    @pytest.mark.asyncio
    async def test_different_result_variable_count(self):
        """Should return NotEquivalentResult without spawning cvc5."""
        expr1 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )
        expr2 = DRCExpression(
            result_variables=[
                ColumnVariable(name="x"),
                ColumnVariable(name="y"),
            ],
            condition=VariableRefNode(name="x"),
        )

        result = await check_equivalence(expr1, expr2)

        assert isinstance(result, NotEquivalentResult)
        assert result.status == "not_equivalent"

    @pytest.mark.asyncio
    async def test_empty_vs_nonempty_result_variables(self):
        """Empty vs non-empty result variables should be not equivalent."""
        expr1 = DRCExpression(
            result_variables=[],
            condition=VariableRefNode(name="x"),
        )
        expr2 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )

        result = await check_equivalence(expr1, expr2)

        assert isinstance(result, NotEquivalentResult)


class TestEquivalenceCheckerWithMockSubprocess:
    """Test equivalence checker with mocked cvc5 subprocess."""

    @pytest.mark.asyncio
    async def test_equivalent_expressions(self):
        """When cvc5 returns 'unsat' for negated equivalence, result is equivalent."""
        expr1 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="x"),
                right=LiteralNode(value=1, data_type="number"),
            ),
        )
        expr2 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="x"),
                right=LiteralNode(value=1, data_type="number"),
            ),
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"unsat", b""))
        mock_proc.returncode = 0

        with patch(
            "text_to_sql_planner.equivalence.equivalence_checker.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await check_equivalence(expr1, expr2)

        assert isinstance(result, EquivalentResult)
        assert result.status == "equivalent"

    @pytest.mark.asyncio
    async def test_not_equivalent_expressions(self):
        """When cvc5 returns 'sat' for negated equivalence, result is not equivalent."""
        expr1 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="x"),
                right=LiteralNode(value=1, data_type="number"),
            ),
        )
        expr2 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="x"),
                right=LiteralNode(value=2, data_type="number"),
            ),
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"sat", b""))
        mock_proc.returncode = 0

        with patch(
            "text_to_sql_planner.equivalence.equivalence_checker.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await check_equivalence(expr1, expr2)

        assert isinstance(result, NotEquivalentResult)
        assert result.status == "not_equivalent"

    @pytest.mark.asyncio
    async def test_timeout_returns_indeterminate(self):
        """When cvc5 times out, result should be indeterminate."""
        expr1 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )
        expr2 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )

        async def slow_communicate(*args, **kwargs):
            await asyncio.sleep(10)
            return (b"sat", b"")

        mock_proc = AsyncMock()
        mock_proc.communicate = slow_communicate
        mock_proc.returncode = 0

        config = EquivalenceCheckerConfig(timeout_seconds=0.1)

        with patch(
            "text_to_sql_planner.equivalence.equivalence_checker.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await check_equivalence(expr1, expr2, config)

        assert isinstance(result, IndeterminateResult)
        assert result.status == "indeterminate"
        assert "imeout" in result.reason or "indeterminate" in result.reason.lower() or result.reason != ""

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_returns_indeterminate(self):
        """When cvc5 exits with non-zero code, result should be indeterminate."""
        expr1 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )
        expr2 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"error", b""))
        mock_proc.returncode = 1

        with patch(
            "text_to_sql_planner.equivalence.equivalence_checker.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await check_equivalence(expr1, expr2)

        assert isinstance(result, IndeterminateResult)
        assert "exit" in result.reason.lower() or "code" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_unparseable_output_returns_indeterminate(self):
        """When cvc5 returns unexpected output, result should be indeterminate."""
        expr1 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )
        expr2 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"unknown\n(reason timeout)", b"")
        )
        mock_proc.returncode = 0

        with patch(
            "text_to_sql_planner.equivalence.equivalence_checker.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = await check_equivalence(expr1, expr2)

        assert isinstance(result, IndeterminateResult)

    @pytest.mark.asyncio
    async def test_same_result_variable_count_proceeds(self):
        """Same result variable count should proceed to cvc5 check."""
        expr1 = DRCExpression(
            result_variables=[ColumnVariable(name="x"), ColumnVariable(name="y")],
            condition=MembershipNode(variables=["x", "y"], relation="R"),
        )
        expr2 = DRCExpression(
            result_variables=[ColumnVariable(name="a"), ColumnVariable(name="b")],
            condition=MembershipNode(variables=["a", "b"], relation="R"),
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"unsat", b""))
        mock_proc.returncode = 0

        with patch(
            "text_to_sql_planner.equivalence.equivalence_checker.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            result = await check_equivalence(expr1, expr2)

        # Should have called cvc5 (subprocess was invoked)
        assert mock_exec.called
        assert isinstance(result, EquivalentResult)

    @pytest.mark.asyncio
    async def test_config_cvc5_path_used(self):
        """The configured cvc5 path should be passed to subprocess."""
        expr1 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )
        expr2 = DRCExpression(
            result_variables=[ColumnVariable(name="x")],
            condition=VariableRefNode(name="x"),
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"sat", b""))
        mock_proc.returncode = 0

        config = EquivalenceCheckerConfig(cvc5_path="/usr/local/bin/cvc5")

        with patch(
            "text_to_sql_planner.equivalence.equivalence_checker.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            await check_equivalence(expr1, expr2, config)

        # Check that the custom path was used
        call_args = mock_exec.call_args_list[0]
        assert call_args[0][0] == "/usr/local/bin/cvc5"
