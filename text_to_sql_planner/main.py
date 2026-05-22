"""Main entry point: orchestrates the text-to-SQL pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

from text_to_sql_planner.converter.question_converter import (
    ConversionSuccess,
    convert_question,
)
from text_to_sql_planner.converter.table_converter import (
    TableConversionSuccess,
    convert_tables,
)
from text_to_sql_planner.planner.llm_client import LLMClientConfig
from text_to_sql_planner.planner.planner import (
    PlannerConfig,
    PlannerSuccess,
    plan,
)
from text_to_sql_planner.sql import SQLSuccess, convert_to_sql
from text_to_sql_planner.types.drc import DRCExpression
from text_to_sql_planner.types.errors import ErrorCode
from text_to_sql_planner.types.operation_tree import OperationTree


@dataclass
class TextToSQLSuccess:
    """Successful text-to-SQL result."""

    sql: str
    operation_tree: OperationTree
    target_expression: DRCExpression


@dataclass
class TextToSQLFailure:
    """Failed text-to-SQL result."""

    error: str
    code: ErrorCode


TextToSQLResult = Union[TextToSQLSuccess, TextToSQLFailure]


# Regex to detect at least one CREATE TABLE statement
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE", re.IGNORECASE
)


async def run(
    question: str,
    schema: str,
    config: PlannerConfig | None = None,
) -> TextToSQLResult:
    """Convert a natural language question into a SQL SELECT statement.

    Orchestrates the full pipeline:
    1. Validate inputs
    2. Convert question to DRC expression
    3. Convert schema tables to DRC relations
    4. Run the planner to build an operation tree
    5. Convert the operation tree to SQL

    Args:
        question: The natural language question.
        schema: The database schema (CREATE TABLE statements).
        config: Planner configuration (optional).

    Returns:
        TextToSQLSuccess with the SQL and operation tree, or
        TextToSQLFailure with an error description.
    """
    if config is None:
        config = PlannerConfig()

    # --- Input validation ---
    if not question or not question.strip():
        return TextToSQLFailure(
            error="Question must not be empty.",
            code=ErrorCode.EMPTY_QUESTION,
        )

    if not schema or not _CREATE_TABLE_RE.search(schema):
        return TextToSQLFailure(
            error="Schema must contain at least one CREATE TABLE statement.",
            code=ErrorCode.INVALID_SCHEMA,
        )

    # --- Step 1: Convert question to DRC ---
    question_result = await convert_question(
        question=question.strip(),
        schema=schema,
        config=config.llm_config,
    )

    if not isinstance(question_result, ConversionSuccess):
        return TextToSQLFailure(
            error=f"Failed to convert question to DRC: {question_result.message}",
            code=ErrorCode.QUESTION_CONVERSION_FAILED,
        )

    target_expression = question_result.expression

    # --- Step 2: Convert tables to DRC relations ---
    table_result = convert_tables(schema)

    if not isinstance(table_result, TableConversionSuccess):
        return TextToSQLFailure(
            error=f"Failed to convert tables: {table_result.error}",
            code=ErrorCode.TABLE_CONVERSION_FAILED,
        )

    if not table_result.relations:
        return TextToSQLFailure(
            error="No tables found in schema.",
            code=ErrorCode.NO_TABLES_FOUND,
        )

    # --- Step 3: Run the planner ---
    planner_result = await plan(
        table_relations=table_result.relations,
        target_relation=target_expression,
        config=config,
    )

    if not isinstance(planner_result, PlannerSuccess):
        error_code = (
            ErrorCode.MAX_ITERATIONS_EXCEEDED
            if planner_result.error_type == "max_iterations_exceeded"
            else ErrorCode.OPERATOR_SELECTION_FAILED
        )
        return TextToSQLFailure(
            error=f"Planning failed: {planner_result.message}",
            code=error_code,
        )

    # --- Step 4: Convert operation tree to SQL ---
    sql_result = convert_to_sql(planner_result.operation_tree, result_variables=target_expression.result_variables)

    if not isinstance(sql_result, SQLSuccess):
        return TextToSQLFailure(
            error=f"SQL conversion failed: {sql_result.error}",
            code=ErrorCode.SQL_CONVERSION_FAILED,
        )

    return TextToSQLSuccess(
        sql=sql_result.sql,
        operation_tree=planner_result.operation_tree,
        target_expression=target_expression,
    )
