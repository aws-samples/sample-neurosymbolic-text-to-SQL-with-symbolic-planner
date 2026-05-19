"""Tests for the main entry point (input validation and orchestration)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from text_to_sql_planner.main import (
    run,
    TextToSQLSuccess,
    TextToSQLFailure,
)
from text_to_sql_planner.types.errors import ErrorCode
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    MembershipNode,
)
from text_to_sql_planner.types.operation_tree import OperationTree, TableLeafNode


# --- Input Validation Tests ---


@pytest.mark.asyncio
async def test_empty_question_returns_error():
    """Empty question should return EMPTY_QUESTION error."""
    schema = "CREATE TABLE users (id INT, name VARCHAR);"
    result = await run(question="", schema=schema)

    assert isinstance(result, TextToSQLFailure)
    assert result.code == ErrorCode.EMPTY_QUESTION
    assert "empty" in result.error.lower()


@pytest.mark.asyncio
async def test_whitespace_only_question_returns_error():
    """Whitespace-only question should return EMPTY_QUESTION error."""
    schema = "CREATE TABLE users (id INT, name VARCHAR);"
    result = await run(question="   \t\n  ", schema=schema)

    assert isinstance(result, TextToSQLFailure)
    assert result.code == ErrorCode.EMPTY_QUESTION


@pytest.mark.asyncio
async def test_schema_with_no_create_table_returns_error():
    """Schema without CREATE TABLE should return INVALID_SCHEMA error."""
    result = await run(
        question="Find all users",
        schema="SELECT * FROM users;",
    )

    assert isinstance(result, TextToSQLFailure)
    assert result.code == ErrorCode.INVALID_SCHEMA
    assert "CREATE TABLE" in result.error


@pytest.mark.asyncio
async def test_empty_schema_returns_error():
    """Empty schema should return INVALID_SCHEMA error."""
    result = await run(question="Find all users", schema="")

    assert isinstance(result, TextToSQLFailure)
    assert result.code == ErrorCode.INVALID_SCHEMA


@pytest.mark.asyncio
async def test_none_like_schema_returns_error():
    """Schema with only whitespace should return INVALID_SCHEMA error."""
    result = await run(question="Find all users", schema="   ")

    assert isinstance(result, TextToSQLFailure)
    assert result.code == ErrorCode.INVALID_SCHEMA


# --- Orchestration Tests (with mocked LLM) ---


@pytest.mark.asyncio
async def test_valid_inputs_with_mocked_planner():
    """Valid inputs with mocked planner should return SQL success."""
    schema = "CREATE TABLE employees (id INT, name VARCHAR, dept_id INT);"
    question = "Find all employees"

    # Create a mock target expression
    mock_expression = DRCExpression(
        result_variables=[
            ColumnVariable(name="id"),
            ColumnVariable(name="name"),
            ColumnVariable(name="dept_id"),
        ],
        condition=MembershipNode(
            variables=["id", "name", "dept_id"],
            relation="employees",
        ),
    )

    # Mock the question converter
    from text_to_sql_planner.converter.question_converter import ConversionSuccess

    mock_conversion = ConversionSuccess(
        expression=mock_expression,
        lisp_syntax="(drc (id name dept_id) (in (id name dept_id) employees))",
    )

    # Mock the planner to return a simple tree
    from text_to_sql_planner.planner.planner import PlannerSuccess

    mock_tree = OperationTree(
        root=TableLeafNode(
            table_name="employees",
            columns=["id", "name", "dept_id"],
            expression=mock_expression,
        )
    )
    mock_planner_result = PlannerSuccess(operation_tree=mock_tree, iterations=1)

    with patch(
        "text_to_sql_planner.main.convert_question",
        new_callable=AsyncMock,
        return_value=mock_conversion,
    ), patch(
        "text_to_sql_planner.main.plan",
        new_callable=AsyncMock,
        return_value=mock_planner_result,
    ):
        result = await run(question=question, schema=schema)

    assert isinstance(result, TextToSQLSuccess)
    assert result.sql  # Should have some SQL output
    assert result.operation_tree is not None
    assert result.target_expression is not None


@pytest.mark.asyncio
async def test_question_conversion_failure_returns_error():
    """If question conversion fails, should return QUESTION_CONVERSION_FAILED."""
    schema = "CREATE TABLE users (id INT, name VARCHAR);"
    question = "Find all users"

    from text_to_sql_planner.converter.question_converter import ConversionError

    mock_error = ConversionError(
        message="LLM returned invalid syntax",
        attempts=3,
        last_raw_output="(invalid",
    )

    with patch(
        "text_to_sql_planner.main.convert_question",
        new_callable=AsyncMock,
        return_value=mock_error,
    ):
        result = await run(question=question, schema=schema)

    assert isinstance(result, TextToSQLFailure)
    assert result.code == ErrorCode.QUESTION_CONVERSION_FAILED
