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
    column_types: dict[str, str] = field(default_factory=dict)  # col_name -> "Int" | "String"
    # Maps sanitised DRC variable name → original SQL column name.
    # Only populated for columns whose original name is not a valid
    # bare identifier (contains spaces, dashes, parens, etc.). The
    # SQL converter uses this to emit backtick-quoted originals.
    original_column_names: dict[str, str] = field(default_factory=dict)


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

# Regex matching valid bare identifiers (no quoting needed).
_BARE_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


def _sanitise_column_name(name: str) -> str:
    """Convert a raw SQL column name into a valid DRC variable identifier.

    DRC variables are plain identifiers matching ``[A-Za-z_][A-Za-z0-9_]*``.
    Column names extracted from BIRD schemas can contain spaces, dashes,
    parentheses, percent signs, and other characters that the DRC parser
    / S-expression syntax cannot represent as bare atoms. This function
    replaces each run of non-identifier characters with a single
    underscore, strips leading/trailing underscores, and ensures the
    result starts with a letter or underscore.

    Examples:
        "Charter Funding Type"  → "Charter_Funding_Type"
        "Percent (%) Eligible"  → "Percent_Eligible"
        "2013-14 CALPADS ..."   → "_2013_14_CALPADS_..."
        "aCL IgG"               → "aCL_IgG"
        "T-CHO"                 → "T_CHO"
        "ANA Pattern"           → "ANA_Pattern"
    """
    if _BARE_IDENT_RE.match(name):
        return name
    # Replace each run of non-word characters with a single underscore.
    sanitised = re.sub(r"[^\w]+", "_", name)
    # Strip leading/trailing underscores.
    sanitised = sanitised.strip("_")
    # If the result starts with a digit, prefix with underscore.
    if sanitised and sanitised[0].isdigit():
        sanitised = f"_{sanitised}"
    # Edge case: completely empty after sanitisation (unlikely but defensive).
    if not sanitised:
        sanitised = "_col"
    return sanitised


def _extract_column_name(definition: str) -> str | None:
    """Extract column name from a column definition string.

    Returns the column name or None if this is a table-level constraint.

    Handles three quoting styles:

    * Bare identifier: ``my_col INTEGER`` → ``my_col``
    * Backtick-quoted: ``` `My Col` INTEGER ``` → ``My Col``
    * Double-quoted: ``"My Col" INTEGER`` → ``My Col``
    * Single-quoted: ``'My Col' INTEGER`` → ``My Col``

    For quoted forms, the name can contain *any* characters except the
    quote — spaces, dashes, dots, parentheses are all preserved
    verbatim. The previous implementation only captured ``\\w+`` even
    inside quotes, so ``` `Examination Date` `` collapsed to
    ``Examination`` and the run-15 dev_1298 schema (with three
    ``aCL Ig*`` columns and an ``ANA Pattern`` column distinct from
    ``ANA``) produced duplicate ``aCL`` and ``ANA`` columns. The
    duplicates corrupted the DRC's column list, made the planner
    pick the wrong slot when filtering, and indirectly drove the
    LLM into the ``OPERATOR_SELECTION_FAILED`` retry loop.
    """
    definition = definition.strip()
    if not definition:
        return None

    # Skip table-level constraints
    if _TABLE_CONSTRAINT_RE.match(definition):
        return None

    # Quoted column name: capture everything up to the matching quote.
    if definition[0] in "`\"'":
        quote = definition[0]
        end = definition.find(quote, 1)
        if end == -1:
            return None  # malformed — unterminated quote
        return definition[1:end]

    # Bare identifier: word characters only.
    match = re.match(r"(\w+)", definition)
    if match:
        return match.group(1)
    return None


# SQL types that map to String in SMT-LIB
_STRING_TYPES = re.compile(
    r"^(VARCHAR|CHAR|TEXT|CLOB|NVARCHAR|NCHAR|NTEXT|STRING)\b",
    re.IGNORECASE,
)


def _extract_column_type(definition: str) -> str:
    """Extract the SMT-LIB sort (Int or String) from a column definition.

    Returns "String" for VARCHAR/CHAR/TEXT types, "Int" for everything else.
    """
    definition = definition.strip()
    # Skip the column name (first token)
    match = re.match(r"[`\"']?\w+[`\"']?\s+(.+)", definition)
    if not match:
        return "Int"
    type_str = match.group(1).strip()
    if _STRING_TYPES.match(type_str):
        return "String"
    return "Int"


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


def _strip_sql_comments(text: str) -> str:
    """Remove SQL comments from text.

    Handles:
    - Single-line comments: -- ... to end of line
    - Block comments: /* ... */
    """
    # Remove block comments first
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Remove single-line comments
    text = re.sub(r"--[^\n]*", "", text)
    return text


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

    # Strip SQL comments before parsing
    schema = _strip_sql_comments(schema)

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
        column_types: dict[str, str] = {}
        original_column_names: dict[str, str] = {}

        for part in parts:
            col_name = _extract_column_name(part)
            if col_name is not None:
                sanitised = _sanitise_column_name(col_name)
                # Handle duplicate sanitised names (rare but possible if
                # e.g. "A B" and "A-B" both map to "A_B"). Disambiguate
                # with a numeric suffix.
                base = sanitised
                counter = 2
                while sanitised in column_types:
                    sanitised = f"{base}_{counter}"
                    counter += 1
                columns.append(sanitised)
                column_types[sanitised] = _extract_column_type(part)
                # Track original name if it differs from the sanitised form.
                if sanitised != col_name:
                    original_column_names[sanitised] = col_name

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
                column_types=column_types,
                original_column_names=original_column_names,
            )
        )

    return TableConversionSuccess(relations=relations)
