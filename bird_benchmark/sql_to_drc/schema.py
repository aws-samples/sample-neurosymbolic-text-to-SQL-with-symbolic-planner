"""Schema parser for SQLite type affinity.

This module reads a CREATE-TABLE-style schema string (the kind
:class:`bird_benchmark.loader.BirdLoader` produces by joining every row of
``sqlite_master.sql``) and produces a column-type lookup keyed by
``(table_name, column_name)`` (both lowercased so callers can do
case-insensitive resolution).

Three column types are recognised, matching the design's Supported_SQL_Subset
narrowing of SQLite's type affinity rules:

- ``"Int"``   — declared type contains ``INT`` (case-insensitive). Covers
  ``INTEGER``, ``INT``, ``BIGINT``, ``SMALLINT``, ``MEDIUMINT``, ``TINYINT``,
  ``INT2``, ``INT8``, etc.
- ``"Real"``  — declared type contains ``REAL``, ``FLOAT``, or ``DOUB``
  (case-insensitive). ``DOUBLE`` and ``DOUBLE PRECISION`` both contain
  ``DOUB``, so multi-word types still classify correctly without a separate
  rule.
- ``"String"`` — anything else, including ``TEXT``, ``VARCHAR(...)``,
  ``CHAR(...)``, ``CLOB``, ``BLOB``, ``DATE``, ``DATETIME``, ``NUMERIC``,
  ``DECIMAL(...)``, and the empty-type case (a column declared without a
  type).

The point of this module is *not* to be a complete SQLite parser. The
translator only needs the affinity bucket so it can resolve ``/`` to integer
or floating-point division (Req 13.5 / 13.6) and so it can decide whether a
double-quoted token is an identifier or a string literal (Req 13.2 / 13.3).
A regex-driven pass that handles the shapes BIRD actually emits is
sufficient. Anything we cannot classify falls through to ``"String"``, which
is also SQLite's affinity default for unrecognised declared types.

Public symbols
--------------
- :data:`ColumnType` — the ``Literal["Int", "Real", "String"]`` alias.
- :func:`parse_schema` — schema string → ``{(table, column): ColumnType}``.
- :func:`affinity_for_declared_type` — declared-type token → ``ColumnType``.
- :func:`numeric_literal_type` — numeric literal token → ``ColumnType``
  (always ``"Int"`` or ``"Real"``).
- :func:`lookup` — case-insensitive lookup with optional table qualifier.
"""

from __future__ import annotations

import re
from typing import Literal


ColumnType = Literal["Int", "Real", "String"]


# --- Public type-classification helpers -----------------------------------


def affinity_for_declared_type(declared: str) -> ColumnType:
    """Classify a declared SQL type by SQLite affinity rules.

    The check is performed against the upper-cased input so it is
    case-insensitive. Type arguments such as ``VARCHAR(50)`` or
    ``DECIMAL(10,2)`` may be passed in: this function treats the entire
    string as a single haystack, and the substrings it looks for
    (``INT``, ``REAL``, ``FLOAT``, ``DOUB``) do not appear inside numeric
    sizes, so passing the pre-stripped form is fine too. Multi-word types
    such as ``DOUBLE PRECISION`` are also accepted as-is.

    The ``INT`` check runs first to match SQLite's own rule that
    ``INTEGER`` always wins over ``REAL``-style hints when both could
    apply.
    """

    declared_upper = declared.upper()
    if "INT" in declared_upper:
        return "Int"
    if "REAL" in declared_upper or "FLOAT" in declared_upper or "DOUB" in declared_upper:
        return "Real"
    return "String"


def numeric_literal_type(token_text: str) -> ColumnType:
    """Classify a numeric literal token by its source shape.

    A literal is :data:`"Real"` if the source spelling contains a decimal
    point or an exponent marker (``e`` or ``E``); otherwise it is
    :data:`"Int"`. The translator uses this to type ``NUMBER`` tokens from
    the lexer when there is no schema column to look up against.
    """

    if "." in token_text or "e" in token_text or "E" in token_text:
        return "Real"
    return "Int"


# --- CREATE TABLE parser ---------------------------------------------------


# Match the *header* of ``CREATE TABLE [IF NOT EXISTS] <name> (`` and stop
# just past the opening paren. The body is then extracted by walking
# characters and tracking parenthesis depth (with single/double/backtick
# quote awareness) so types like ``VARCHAR(50)`` and ``DECIMAL(10,2)`` do
# not prematurely close the column block. The final trailing ``;`` after
# the close paren is optional.
_CREATE_TABLE_HEAD_RE = re.compile(
    r"""
    CREATE \s+ TABLE \s+
    (?: IF \s+ NOT \s+ EXISTS \s+ )?
    (?: " (?P<dq_name> [^"]+ ) "
      | ` (?P<bt_name> [^`]+ ) `
      | (?P<bare_name> \w+ )
    )
    \s* \(
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _extract_balanced_body(text: str, open_paren_index: int) -> tuple[str, int] | None:
    """Walk ``text`` from just after an opening ``(`` to the matching ``)``.

    Returns ``(body, end_index)`` where ``body`` is the substring between
    the parens (exclusive on both ends) and ``end_index`` is the position
    of the closing ``)`` (so the caller can resume scanning at
    ``end_index + 1``). Returns :data:`None` if the parens are
    unbalanced. Single-quoted, double-quoted, and backtick-quoted regions
    are tracked so quoted parens do not affect the depth count, and
    SQLite-style doubled quote escapes inside literals are handled.
    """

    depth = 1
    i = open_paren_index
    in_single = False
    in_double = False
    in_backtick = False
    while i < len(text):
        ch = text[i]
        if in_single:
            if ch == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
        elif in_double:
            if ch == '"':
                if i + 1 < len(text) and text[i + 1] == '"':
                    i += 2
                    continue
                in_double = False
        elif in_backtick:
            if ch == "`":
                in_backtick = False
        else:
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == "`":
                in_backtick = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[open_paren_index:i], i
        i += 1
    return None


# Constraint-style lines that share the column block but are not column
# definitions. The first whitespace-delimited token of a column-block entry
# is checked against this set (case-insensitive); matches are skipped.
_CONSTRAINT_KEYWORDS: frozenset[str] = frozenset(
    {
        "PRIMARY",
        "FOREIGN",
        "UNIQUE",
        "CHECK",
        "CONSTRAINT",
        "KEY",  # standalone KEY (...) appears in some MySQL-flavoured dumps
        "INDEX",
    }
)


def _split_top_level_commas(body: str) -> list[str]:
    """Split a CREATE TABLE body on commas at parenthesis depth zero.

    ``DECIMAL(10,2)`` and ``CHECK(x IN (1,2,3))`` both contain commas
    inside parentheses; those must not split the column list. Single- and
    double-quoted regions are also tracked so quoted identifiers/literals
    containing commas don't accidentally split.
    """

    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    in_backtick = False

    i = 0
    while i < len(body):
        ch = body[i]
        if in_single:
            buf.append(ch)
            if ch == "'":
                # SQLite-style '' doubling stays inside the literal.
                if i + 1 < len(body) and body[i + 1] == "'":
                    buf.append("'")
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if in_double:
            buf.append(ch)
            if ch == '"':
                if i + 1 < len(body) and body[i + 1] == '"':
                    buf.append('"')
                    i += 2
                    continue
                in_double = False
            i += 1
            continue
        if in_backtick:
            buf.append(ch)
            if ch == "`":
                in_backtick = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            buf.append(ch)
        elif ch == '"':
            in_double = True
            buf.append(ch)
        elif ch == "`":
            in_backtick = True
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            if depth > 0:
                depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1

    if buf:
        parts.append("".join(buf))
    return parts


def _strip_identifier_quotes(token: str) -> str:
    """Strip a single layer of ``"..."`` or ``\\`...\\``` quoting.

    Returns the token unchanged if it is not quoted.
    """

    token = token.strip()
    if len(token) >= 2:
        if token[0] == '"' and token[-1] == '"':
            return token[1:-1].replace('""', '"')
        if token[0] == "`" and token[-1] == "`":
            return token[1:-1].replace("``", "`")
    return token


def _parse_column_definition(definition: str) -> tuple[str, ColumnType] | None:
    """Parse a single column definition into ``(column_name, ColumnType)``.

    Returns ``None`` for table-level constraint clauses (``PRIMARY KEY (...)``,
    ``FOREIGN KEY ...``, ``UNIQUE (...)``, ``CHECK (...)``, ``CONSTRAINT
    name ...``) so the caller can skip them.
    """

    text = definition.strip()
    if not text:
        return None

    # Detect quoted column name first, since constraint keywords cannot
    # appear quoted.
    if text[0] == '"':
        end = text.find('"', 1)
        # Walk past any "" doubling.
        while end != -1 and end + 1 < len(text) and text[end + 1] == '"':
            end = text.find('"', end + 2)
        if end == -1:
            return None
        name_raw = text[: end + 1]
        rest = text[end + 1 :].strip()
        column_name = _strip_identifier_quotes(name_raw)
    elif text[0] == "`":
        end = text.find("`", 1)
        if end == -1:
            return None
        name_raw = text[: end + 1]
        rest = text[end + 1 :].strip()
        column_name = _strip_identifier_quotes(name_raw)
    else:
        # Bare identifier: split off the first whitespace-delimited token.
        m = re.match(r"\s*(\w+)\s*(.*)$", text, re.DOTALL)
        if m is None:
            return None
        first_token = m.group(1)
        if first_token.upper() in _CONSTRAINT_KEYWORDS:
            return None
        column_name = first_token
        rest = m.group(2).strip()

    # ``rest`` now contains the type words (and possibly trailing
    # constraints like ``NOT NULL DEFAULT 0``). Pull off the type portion:
    # one or more leading ``\w+`` tokens, optionally followed by ``(...)``
    # type arguments. ``DOUBLE PRECISION`` is two words, so we collect
    # consecutive identifier tokens until we hit either an open paren
    # (type args) or a non-type keyword like NOT/DEFAULT/PRIMARY.
    declared = _extract_declared_type(rest)
    if declared == "":
        # No type at all → SQLite affinity rule defaults to BLOB/TEXT;
        # we model that as ``"String"``.
        return column_name, "String"

    return column_name, affinity_for_declared_type(declared)


# Tokens that terminate the type-words run inside a column definition.
# Anything in this set marks the start of constraints rather than more
# type words. ``COLLATE`` is included because it accepts an identifier
# argument that we must not concatenate into the declared type.
_TYPE_TERMINATORS: frozenset[str] = frozenset(
    {
        "NOT",
        "NULL",
        "DEFAULT",
        "PRIMARY",
        "KEY",
        "UNIQUE",
        "CHECK",
        "REFERENCES",
        "COLLATE",
        "GENERATED",
        "AS",
        "ON",
        "AUTOINCREMENT",
        "CONSTRAINT",
    }
)


def _extract_declared_type(rest: str) -> str:
    """Pull the declared-type words off the front of a column definition.

    Handles the three shapes in the design:

    - ``column_name TYPE_NAME`` (e.g. ``id INTEGER``).
    - ``column_name TYPE_NAME(args)`` (e.g. ``name VARCHAR(50)``,
      ``price DECIMAL(10,2)``).
    - Multi-word types (e.g. ``DOUBLE PRECISION``).

    Returns the declared-type substring (with its parenthesised args
    stripped, since affinity classification only needs the words).
    """

    rest = rest.strip()
    if not rest:
        return ""

    parts: list[str] = []
    i = 0
    while i < len(rest):
        # Skip leading whitespace.
        while i < len(rest) and rest[i].isspace():
            i += 1
        if i >= len(rest):
            break
        ch = rest[i]
        if ch == "(":
            # Type argument list. Skip to the matching close-paren and
            # stop collecting type words afterwards: ``DECIMAL(10,2) NOT
            # NULL`` should not pull NOT/NULL into the type.
            depth = 1
            i += 1
            while i < len(rest) and depth > 0:
                if rest[i] == "(":
                    depth += 1
                elif rest[i] == ")":
                    depth -= 1
                i += 1
            break
        if not (ch.isalnum() or ch == "_"):
            # Hit punctuation outside an identifier — stop.
            break
        # Read a word token.
        j = i
        while j < len(rest) and (rest[j].isalnum() or rest[j] == "_"):
            j += 1
        word = rest[i:j]
        if word.upper() in _TYPE_TERMINATORS:
            break
        parts.append(word)
        i = j

    return " ".join(parts)


def parse_schema(schema: str) -> dict[tuple[str, str], ColumnType]:
    """Parse a CREATE-TABLE schema string into a column-type lookup.

    The result maps ``(table_name_lower, column_name_lower)`` tuples to a
    :data:`ColumnType` so callers can do case-insensitive resolution
    against unquoted SQL identifiers and against double-quoted tokens
    that SQLite treats case-insensitively. Tables and columns the parser
    cannot recognise are silently dropped — the translator already falls
    back to ``"String"`` for unknown columns.
    """

    result: dict[tuple[str, str], ColumnType] = {}
    if not schema:
        return result

    pos = 0
    while pos < len(schema):
        match = _CREATE_TABLE_HEAD_RE.search(schema, pos)
        if match is None:
            break
        table_name_raw = (
            match.group("dq_name")
            or match.group("bt_name")
            or match.group("bare_name")
        )
        body_start = match.end()  # just past the "("
        balanced = _extract_balanced_body(schema, body_start)
        if balanced is None:
            # Unbalanced parens: stop scanning rather than risk pulling
            # garbage from later CREATE TABLE blocks.
            break
        body, close_index = balanced
        pos = close_index + 1
        if not table_name_raw:
            continue
        table_name = table_name_raw.lower()
        for piece in _split_top_level_commas(body):
            parsed = _parse_column_definition(piece)
            if parsed is None:
                continue
            column_name, column_type = parsed
            result[(table_name, column_name.lower())] = column_type

    return result


def lookup(
    schema_map: dict[tuple[str, str], ColumnType],
    table: str | None,
    column: str,
) -> ColumnType | None:
    """Look up a column's type, case-insensitively.

    When ``table`` is supplied, returns the type for that exact
    ``(table, column)`` pair (lowercased) or :data:`None` if the pair is
    not present. When ``table`` is :data:`None`, the lookup succeeds only
    if exactly one table in the schema map has the column; ambiguous and
    missing names both yield :data:`None`. The translator uses the
    ``None`` form when a SQL ``ColumnRef`` lacks an explicit table
    qualifier and there is only one table in scope.
    """

    column_lower = column.lower()
    if table is not None:
        return schema_map.get((table.lower(), column_lower))

    matches: list[ColumnType] = [
        value for (_t, c), value in schema_map.items() if c == column_lower
    ]
    if len(matches) == 1:
        return matches[0]
    return None
