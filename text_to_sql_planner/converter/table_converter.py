"""Table converter: transforms CREATE TABLE statements into DRC expressions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Union

from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    MembershipNode,
)


@dataclass
class TableRelation:
    """A parsed table with its DRC expression."""
    table_name: str
    columns: list[str]
    expression: DRCExpression


@dataclass
class TableConversionSuccess:
    """Successful conversion result containing parsed table relations."""
    relations: list[TableRelation]


@dataclass
class TableConversionFailure:
    """Failed conversion result with error description."""
    error: str


TableConversionResult = Union[TableConversionSuccess, TableConversionFailure]


# Regex to match the CREATE TABLE header (up to the opening paren)
_CREATE_TABLE_HEADER_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:[`\"']?(\w+)[`\"']?\.)?[`\"']?(\w+)[`\"']?"
    r"\s*\(",
    re.IGNORECASE,
)

# Regex to identify table-level constraints that should be skipped
_TABLE_CONSTRAINT_RE = re.compile(
    r"^\s*(?:PRIMARY\s+KEY|UNIQUE|CHECK|FOREIGN\s+KEY|CONSTRAINT|INDEX|KEY)\b",
    re.IGNORECASE,
)


def _extract_column_name(definition: str) -> str | None:
    """Extract column name from a column definition string.

    Returns the column name or None if this is a table-level constraint.
    """
    definition = definition.strip()
    if not definition:
        return None

    # Skip table-level constraints
    if _TABLE_CONSTRAINT_RE.match(definition):
        return None

    # Column name is the first token; may be quoted/backticked
    match = re.match(r"[`\"']?(\w+)[`\"']?", definition)
    if match:
        return match.group(1)
    return None


def _split_column_definitions(body: str) -> list[str]:
    """Split the column definition body by commas, respecting parentheses.

    Handles cases like VARCHAR(10,2) where commas appear inside parens.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []

    for char in body:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)

    # Don't forget the last segment
    if current:
        parts.append("".join(current))

    return parts


def _extract_balanced_parens(text: str, start: int) -> str | None:
    """Extract content between balanced parentheses starting at position `start`.

    `start` should point to the character right after the opening '('.
    Returns the content between the outermost parens, or None if unbalanced.
    """
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    if depth == 0:
        # Return content between the opening and closing parens
        return text[start : i - 1]
    return None


def convert_tables(schema: str) -> TableConversionResult:
    """Convert CREATE TABLE statements into DRC expressions.

    Args:
        schema: A string containing one or more CREATE TABLE statements.

    Returns:
        TableConversionSuccess with a list of TableRelation objects, or
        TableConversionFailure with an error message.
    """
    if not schema or not schema.strip():
        return TableConversionFailure(error="Empty schema: no input provided")

    matches = list(_CREATE_TABLE_HEADER_RE.finditer(schema))

    if not matches:
        return TableConversionFailure(
            error="No valid CREATE TABLE statements found in input"
        )

    relations: list[TableRelation] = []

    for match in matches:
        # Group 1 is optional schema prefix, group 2 is table name
        table_name = match.group(2)
        # The match ends right after the '(', so extract balanced body
        body = _extract_balanced_parens(schema, match.end())
        if body is None:
            return TableConversionFailure(
                error=f"Unbalanced parentheses in CREATE TABLE for '{table_name}'"
            )

        # Parse column definitions
        parts = _split_column_definitions(body)
        columns: list[str] = []

        for part in parts:
            col_name = _extract_column_name(part)
            if col_name is not None:
                columns.append(col_name)

        if not columns:
            return TableConversionFailure(
                error=f"Table '{table_name}' has no column definitions"
            )

        # Build DRC expression
        result_variables = [ColumnVariable(name=col) for col in columns]
        condition = MembershipNode(variables=list(columns), relation=table_name)
        expression = DRCExpression(
            result_variables=result_variables,
            condition=condition,
        )

        relations.append(
            TableRelation(
                table_name=table_name,
                columns=columns,
                expression=expression,
            )
        )

    return TableConversionSuccess(relations=relations)
