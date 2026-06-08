"""Tests for backtick-quoted identifier support in the SQL→DRC pipeline.

Background — run-15 dev_18 / dev_1109:
BIRD's gold queries use SQLite's backtick-quoted identifier syntax for
column names containing spaces or punctuation:

    SELECT T1.`Charter Funding Type` FROM frpm AS T1 WHERE ...
    SELECT t2.`date` FROM Team_Attributes AS t2 WHERE ...

The lexer rejected the leading ``\``` as ``unexpected character '\``'``
(ConverterError parse_error), so the gold side never made it to the
translator and the test cases came back ``skipped``.

The lexer fix mirrors the existing double-quoted path: an opening
backtick begins a quoted identifier whose body runs until the closing
backtick. Doubled backticks (``\`\```) inside the body are unescaped
to a single backtick, matching SQLite's escape convention for the
other quote characters.

Unlike double-quoted tokens, backtick-quoted identifiers are
unambiguously identifiers — there's no string-fallback resolution
under SQLite's rules — so they emit ``Token(IDENT, ..., quoted=False)``.
"""

from __future__ import annotations

from bird_benchmark.sql_to_drc import convert_sql
from bird_benchmark.sql_to_drc.lexer import tokenize, TokenKind
from bird_benchmark.types import ConverterError


# ---------------------------------------------------------------------------
# Lexer-level tests
# ---------------------------------------------------------------------------


def test_lexer_accepts_backtick_quoted_identifier():
    """A bare ``\`name\``` lexes as a single IDENT token whose text
    is the body without surrounding backticks."""
    tokens = tokenize("`name`")
    # First token is the identifier; second is EOF.
    assert tokens[0].kind == TokenKind.IDENT
    assert tokens[0].text == "name"
    # Backtick form is unambiguously an identifier; quoted should
    # NOT be set (that flag means "double-quoted ambiguous form").
    assert tokens[0].quoted is False


def test_lexer_accepts_backtick_quoted_identifier_with_space():
    """``\`Charter Funding Type\``` survives as a single IDENT token
    with the embedded space preserved."""
    tokens = tokenize("`Charter Funding Type`")
    assert tokens[0].kind == TokenKind.IDENT
    assert tokens[0].text == "Charter Funding Type"


def test_lexer_accepts_backtick_quoted_identifier_with_dash():
    """``\`T-CHO\``` keeps the dash in the body."""
    tokens = tokenize("`T-CHO`")
    assert tokens[0].kind == TokenKind.IDENT
    assert tokens[0].text == "T-CHO"


def test_lexer_unescapes_doubled_backticks():
    """``\`a\`\`b\``` is the identifier ``a\`b`` (single backtick in
    the middle), mirroring how the existing double-quoted handler
    treats ``""``."""
    tokens = tokenize("`a``b`")
    assert tokens[0].kind == TokenKind.IDENT
    assert tokens[0].text == "a`b"


def test_lexer_unterminated_backtick_emits_lex_error():
    """An unclosed backtick produces a structured LEX_ERROR token
    rather than crashing or silently absorbing the rest of the input."""
    tokens = tokenize("`name")
    # First token should be the LEX_ERROR sentinel.
    assert tokens[0].kind == TokenKind.LEX_ERROR
    assert "backtick" in tokens[0].text.lower()


def test_lexer_handles_qualified_backtick_column():
    """``T1.\`Charter Funding Type\``` lexes as IDENT(T1) OP(.)
    IDENT(Charter Funding Type)."""
    tokens = tokenize("T1.`Charter Funding Type`")
    # Strip trailing EOF.
    real = [t for t in tokens if t.kind != TokenKind.EOF]
    assert len(real) == 3
    assert real[0].kind == TokenKind.IDENT and real[0].text == "T1"
    assert real[1].kind == TokenKind.OP and real[1].text == "."
    assert real[2].kind == TokenKind.IDENT
    assert real[2].text == "Charter Funding Type"


# ---------------------------------------------------------------------------
# End-to-end: convert_sql succeeds on the dev_18 / dev_1109 shapes
# ---------------------------------------------------------------------------


_SCHEMA_FRPM = (
    "CREATE TABLE frpm ("
    "  CDSCode TEXT,"
    "  `Charter Funding Type` TEXT,"
    "  `County Name` TEXT"
    ");\n"
    "CREATE TABLE satscores ("
    "  cds TEXT,"
    "  NumTstTakr INTEGER"
    ");"
)


def test_convert_sql_accepts_dev_18_gold_shape():
    """The exact dev_18 shape: backtick-qualified column references
    in the WHERE clause."""
    sql = (
        "SELECT COUNT(T1.CDSCode) "
        "FROM frpm AS T1 INNER JOIN satscores AS T2 "
        "ON T1.CDSCode = T2.cds "
        "WHERE T1.`Charter Funding Type` = 'Directly funded' "
        "AND T1.`County Name` = 'Fresno' "
        "AND T2.NumTstTakr <= 250"
    )
    result = convert_sql(sql, _SCHEMA_FRPM)
    assert not isinstance(result, ConverterError), (
        f"expected success, got {result}"
    )


_SCHEMA_TEAM = (
    "CREATE TABLE Team ("
    "  team_api_id INTEGER,"
    "  team_short_name TEXT"
    ");\n"
    "CREATE TABLE Team_Attributes ("
    "  team_api_id INTEGER,"
    "  `date` TEXT,"
    "  buildUpPlayDribblingClass TEXT"
    ");"
)


def test_convert_sql_accepts_dev_1109_gold_shape():
    """The dev_1109 shape: a backtick-quoted ``\`date\``` column. The
    inner ``SUBSTR(...)`` function is parsed as a generic function
    call (the converter doesn't need to know its semantics — it's
    treated as an uninterpreted function head, just like STRFTIME)."""
    sql = (
        "SELECT t2.buildUpPlayDribblingClass "
        "FROM Team AS t1 INNER JOIN Team_Attributes AS t2 "
        "ON t1.team_api_id = t2.team_api_id "
        "WHERE t1.team_short_name = 'LEI' "
        "AND SUBSTR(t2.`date`, 1, 10) = '2015-09-10'"
    )
    result = convert_sql(sql, _SCHEMA_TEAM)
    assert not isinstance(result, ConverterError), (
        f"expected success, got {result}"
    )


def test_convert_sql_backtick_column_resolves_against_schema():
    """A backtick-qualified column in the SELECT list resolves against
    the schema's matching backtick-defined column."""
    sql = "SELECT T1.`County Name` FROM frpm AS T1 WHERE T1.CDSCode = '01'"
    result = convert_sql(sql, _SCHEMA_FRPM)
    assert not isinstance(result, ConverterError)
