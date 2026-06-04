"""SQL lexer for the bird_benchmark SQL→DRC converter.

This module turns a SQL source string into a flat list of :class:`Token`
records. The parser in :mod:`bird_benchmark.sql_to_drc.parser` consumes that
list and decides whether each token is well-formed in context; the lexer
itself is intentionally permissive.

Token kinds
-----------
- ``KEYWORD`` — one of the reserved words listed in :data:`_KEYWORDS`.
  Recognised case-insensitively, but the original source text is preserved
  in :attr:`Token.text` so error messages can quote the SQL verbatim. The
  parser dispatches on the upper-cased form.
- ``IDENT`` — an unquoted identifier matching ``[A-Za-z_][A-Za-z0-9_]*`` or
  a double-quoted token (``"foo"`` or ``"foo""bar"`` for embedded quotes).
  The parser/translator decides whether a double-quoted token resolves as
  an identifier or as a string per Req 13.2 / 13.3, so the lexer just emits
  IDENT for both shapes; the surrounding double quotes are stripped and
  ``""`` doubling is unescaped.
- ``STRING`` — a SQLite single-quoted literal. ``''`` doubling is
  recognised inside the literal. The token text is the unescaped body
  (without the surrounding quotes).
- ``NUMBER`` — an integer or real literal matching
  ``\\d+(\\.\\d+)?([eE][+-]?\\d+)?``. Type affinity (Int vs Real) is the
  parser/translator's job; the lexer just preserves the source spelling.
- ``OP`` — one of ``= != <> < > <= >= + - * / || ( ) , . ;``. The parser
  treats ``<>`` as an alias for ``!=``.
- ``EOF`` — emitted exactly once as the final token in the returned list.
- ``LEX_ERROR`` — sentinel emitted when the source text contains a
  fundamentally malformed run (unterminated single-quoted string,
  unterminated double-quoted token, unterminated ``/* ... */`` block
  comment, or a stray character that is not part of the operator set). The
  parser surfaces these as ``ConverterError(kind="parse_error", ...)``
  so the framework can keep running. After a ``LEX_ERROR`` token the lexer
  appends an ``EOF`` and stops; downstream callers should not rely on any
  further tokens.

Comments
--------
``--`` runs to end of line and ``/* ... */`` block comments may span
multiple lines; both are skipped silently while ``(line, column)`` is
tracked through the consumed characters.

Empty / whitespace-only / comment-only input
--------------------------------------------
Such input emits a single ``EOF`` token at line 1, column 1 (Req 2.7). The
parser converts that into a ``ConverterError(kind="parse_error",
message="no SELECT statement found", line=1, column=1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TokenKind(str, Enum):
    """Token category. Inheriting from ``str`` keeps comparisons easy."""

    KEYWORD = "KEYWORD"
    IDENT = "IDENT"
    STRING = "STRING"
    NUMBER = "NUMBER"
    OP = "OP"
    EOF = "EOF"
    LEX_ERROR = "LEX_ERROR"


@dataclass
class Token:
    """A single lexical token with 1-indexed source coordinates.

    The ``text`` field is the *semantic* value of the token:
      - For ``KEYWORD`` / ``IDENT`` (unquoted) / ``NUMBER`` / ``OP``: the
        original source spelling.
      - For ``STRING``: the unescaped body (no surrounding quotes; ``''``
        collapsed to ``'``).
      - For ``IDENT`` from a double-quoted token: the unescaped body (no
        surrounding quotes; ``""`` collapsed to ``"``).
      - For ``EOF``: the empty string.
      - For ``LEX_ERROR``: a human-readable message describing the
        malformation.

    The ``quoted`` field is ``True`` only for ``IDENT`` tokens that came
    from a SQLite-style double-quoted source token (e.g. ``"name"``). The
    translator uses this to apply Req 13.2 / 13.3: a double-quoted token
    resolves as an identifier only when it matches a name in the active
    schema, and otherwise falls back to a string literal. All other
    token kinds, and unquoted identifiers, set ``quoted = False``.
    """

    kind: TokenKind
    text: str
    line: int
    column: int
    quoted: bool = False


# Reserved words. Recognised case-insensitively against the upper-cased
# token text; the parser also compares the upper-cased form for dispatch.
_KEYWORDS: frozenset[str] = frozenset(
    {
        "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL",
        "OUTER", "CROSS", "ON", "GROUP", "BY", "ORDER", "LIMIT", "AND", "OR",
        "NOT", "IN", "EXISTS", "AS", "ASC", "DESC", "DISTINCT", "HAVING",
        "UNION", "INTERSECT", "EXCEPT", "WITH", "RECURSIVE", "CASE", "WHEN",
        "THEN", "ELSE", "END", "OVER", "PARTITION", "LIKE", "BETWEEN", "IS",
        "NULL", "COUNT", "SUM", "AVG", "MIN", "MAX",
    }
)

# Multi-character operators are matched first (longest match wins).
_MULTI_CHAR_OPS: tuple[str, ...] = ("!=", "<>", "<=", ">=", "||")

# Single-character operators / punctuation.
_SINGLE_CHAR_OPS: frozenset[str] = frozenset("=<>+-*/(),.;")


def tokenize(sql: str) -> list[Token]:
    """Lex ``sql`` into a list of tokens ending in a single ``EOF``.

    The returned list always contains at least one token (the trailing
    ``EOF``). On a lex-level malformation the list contains a
    ``LEX_ERROR`` token at the offending position followed by ``EOF``;
    no further tokens are produced past the error.
    """

    tokens: list[Token] = []
    pos = 0
    line = 1
    col = 1
    n = len(sql)

    def advance(k: int = 1) -> None:
        """Move ``pos`` forward by ``k`` characters, updating ``line``/``col``."""
        nonlocal pos, line, col
        for _ in range(k):
            if pos >= n:
                return
            if sql[pos] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            pos += 1

    while pos < n:
        ch = sql[pos]

        # --- whitespace ----------------------------------------------------
        if ch.isspace():
            advance()
            continue

        # --- line comment: -- to end of line ------------------------------
        if ch == "-" and pos + 1 < n and sql[pos + 1] == "-":
            advance(2)
            while pos < n and sql[pos] != "\n":
                advance()
            # The newline (if any) is left for the whitespace branch above
            # so that ``line``/``col`` advance through it uniformly.
            continue

        # --- block comment: /* ... */ -------------------------------------
        if ch == "/" and pos + 1 < n and sql[pos + 1] == "*":
            start_line, start_col = line, col
            advance(2)
            closed = False
            while pos < n:
                if sql[pos] == "*" and pos + 1 < n and sql[pos + 1] == "/":
                    advance(2)
                    closed = True
                    break
                advance()
            if not closed:
                tokens.append(
                    Token(
                        TokenKind.LEX_ERROR,
                        "unterminated block comment",
                        start_line,
                        start_col,
                    )
                )
                tokens.append(Token(TokenKind.EOF, "", line, col))
                return tokens
            continue

        # --- single-quoted string literal ---------------------------------
        if ch == "'":
            start_line, start_col = line, col
            advance()  # opening quote
            buf: list[str] = []
            closed = False
            while pos < n:
                if sql[pos] == "'":
                    # SQLite-style '' escape inside a string literal.
                    if pos + 1 < n and sql[pos + 1] == "'":
                        buf.append("'")
                        advance(2)
                        continue
                    advance()  # closing quote
                    closed = True
                    break
                buf.append(sql[pos])
                advance()
            if not closed:
                tokens.append(
                    Token(
                        TokenKind.LEX_ERROR,
                        "unterminated string literal",
                        start_line,
                        start_col,
                    )
                )
                tokens.append(Token(TokenKind.EOF, "", line, col))
                return tokens
            tokens.append(
                Token(TokenKind.STRING, "".join(buf), start_line, start_col)
            )
            continue

        # --- double-quoted token (deferred: identifier or string) ---------
        if ch == '"':
            start_line, start_col = line, col
            advance()  # opening quote
            buf = []
            closed = False
            while pos < n:
                if sql[pos] == '"':
                    if pos + 1 < n and sql[pos + 1] == '"':
                        buf.append('"')
                        advance(2)
                        continue
                    advance()  # closing quote
                    closed = True
                    break
                buf.append(sql[pos])
                advance()
            if not closed:
                tokens.append(
                    Token(
                        TokenKind.LEX_ERROR,
                        "unterminated quoted identifier",
                        start_line,
                        start_col,
                    )
                )
                tokens.append(Token(TokenKind.EOF, "", line, col))
                return tokens
            tokens.append(
                Token(TokenKind.IDENT, "".join(buf), start_line, start_col, quoted=True)
            )
            continue

        # --- numeric literal ---------------------------------------------
        # Pattern: \d+(\.\d+)?([eE][+-]?\d+)?
        if ch.isdigit():
            start_line, start_col = line, col
            start_pos = pos
            while pos < n and sql[pos].isdigit():
                advance()
            # Optional fractional part. We require a digit *after* the dot
            # so that tokens like ``t.col`` and ``1.col`` lex as
            # NUMBER + OP(.) + IDENT instead of swallowing the dot.
            if (
                pos < n
                and sql[pos] == "."
                and pos + 1 < n
                and sql[pos + 1].isdigit()
            ):
                advance()  # the dot
                while pos < n and sql[pos].isdigit():
                    advance()
            # Optional exponent: e/E [+/-]? digits. Same look-ahead trick:
            # only consume ``e`` if a valid exponent follows, so that
            # identifiers like ``e1`` aren't mis-eaten.
            if pos < n and sql[pos] in "eE":
                peek = pos + 1
                if peek < n and sql[peek] in "+-":
                    peek += 1
                if peek < n and sql[peek].isdigit():
                    advance()  # e/E
                    if pos < n and sql[pos] in "+-":
                        advance()
                    while pos < n and sql[pos].isdigit():
                        advance()
            tokens.append(
                Token(
                    TokenKind.NUMBER, sql[start_pos:pos], start_line, start_col
                )
            )
            continue

        # --- identifier / keyword ----------------------------------------
        if ch.isalpha() or ch == "_":
            start_line, start_col = line, col
            start_pos = pos
            while pos < n and (sql[pos].isalnum() or sql[pos] == "_"):
                advance()
            text = sql[start_pos:pos]
            kind = (
                TokenKind.KEYWORD
                if text.upper() in _KEYWORDS
                else TokenKind.IDENT
            )
            tokens.append(Token(kind, text, start_line, start_col))
            continue

        # --- multi-character operator ------------------------------------
        matched = False
        for op in _MULTI_CHAR_OPS:
            if sql.startswith(op, pos):
                tokens.append(Token(TokenKind.OP, op, line, col))
                advance(len(op))
                matched = True
                break
        if matched:
            continue

        # --- single-character operator / punctuation ---------------------
        if ch in _SINGLE_CHAR_OPS:
            tokens.append(Token(TokenKind.OP, ch, line, col))
            advance()
            continue

        # --- unrecognised character --------------------------------------
        tokens.append(
            Token(
                TokenKind.LEX_ERROR,
                f"unexpected character {ch!r}",
                line,
                col,
            )
        )
        tokens.append(Token(TokenKind.EOF, "", line, col))
        return tokens

    # Empty / whitespace-only / comment-only input: anchor EOF at (1, 1)
    # per Req 2.7. Otherwise EOF lands at whatever position the scanner
    # advanced to (one past the last consumed character).
    if not tokens:
        tokens.append(Token(TokenKind.EOF, "", 1, 1))
    else:
        tokens.append(Token(TokenKind.EOF, "", line, col))
    return tokens


__all__ = ["Token", "TokenKind", "tokenize"]
