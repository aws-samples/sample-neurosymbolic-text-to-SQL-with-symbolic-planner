"""Lexer for Lisp S-expression DRC syntax.

Tokenizes input strings into a sequence of tokens: LPAREN, RPAREN,
SYMBOL, STRING, NUMBER, and EOF.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator

from text_to_sql_planner.types.errors import ParseError


class TokenType(Enum):
    LPAREN = auto()
    RPAREN = auto()
    SYMBOL = auto()
    STRING = auto()
    NUMBER = auto()
    EOF = auto()


@dataclass(frozen=True, slots=True)
class Token:
    type: TokenType
    value: str
    offset: int

    def __repr__(self) -> str:
        if self.type in (TokenType.SYMBOL, TokenType.STRING, TokenType.NUMBER):
            return f"Token({self.type.name}, {self.value!r}, offset={self.offset})"
        return f"Token({self.type.name}, offset={self.offset})"


# Characters that terminate a symbol/number atom
_DELIMITERS = set(" \t\n\r\f\v()")


class Lexer:
    """Tokenizer for Lisp S-expression syntax.

    Can be used as an iterator or via the `tokenize()` class method
    to get a full list of tokens.
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0
        self._length = len(source)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def tokenize(cls, source: str) -> list[Token]:
        """Tokenize the full source and return a list of tokens (including EOF)."""
        lexer = cls(source)
        tokens: list[Token] = []
        while True:
            token = lexer.next_token()
            tokens.append(token)
            if token.type is TokenType.EOF:
                break
        return tokens

    def __iter__(self) -> Iterator[Token]:
        return self

    def __next__(self) -> Token:
        token = self.next_token()
        if token.type is TokenType.EOF:
            raise StopIteration
        return token

    def next_token(self) -> Token:
        """Return the next token from the source."""
        self._skip_whitespace()

        if self._pos >= self._length:
            return Token(TokenType.EOF, "", self._pos)

        ch = self._source[self._pos]

        if ch == "(":
            token = Token(TokenType.LPAREN, "(", self._pos)
            self._pos += 1
            return token

        if ch == ")":
            token = Token(TokenType.RPAREN, ")", self._pos)
            self._pos += 1
            return token

        if ch == '"':
            return self._read_string()

        # Number detection: starts with digit, or '-' followed by digit
        if ch.isdigit() or (
            ch == "-"
            and self._pos + 1 < self._length
            and self._source[self._pos + 1].isdigit()
        ):
            return self._read_number()

        return self._read_symbol()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _skip_whitespace(self) -> None:
        while self._pos < self._length and self._source[self._pos].isspace():
            self._pos += 1

    def _read_string(self) -> Token:
        """Read a double-quoted string with escape sequences (\\\\ and \\\")."""
        start = self._pos
        self._pos += 1  # skip opening quote
        chars: list[str] = []

        while self._pos < self._length:
            ch = self._source[self._pos]
            if ch == "\\":
                self._pos += 1
                if self._pos >= self._length:
                    raise _make_error(
                        start, "Unterminated escape sequence in string", self._source
                    )
                escaped = self._source[self._pos]
                if escaped == '"':
                    chars.append('"')
                elif escaped == "\\":
                    chars.append("\\")
                else:
                    # Preserve unknown escapes as-is
                    chars.append("\\")
                    chars.append(escaped)
                self._pos += 1
            elif ch == '"':
                self._pos += 1  # skip closing quote
                return Token(TokenType.STRING, "".join(chars), start)
            else:
                chars.append(ch)
                self._pos += 1

        raise _make_error(start, "Unterminated string literal", self._source)

    def _read_number(self) -> Token:
        """Read an integer or float literal."""
        start = self._pos
        # Consume optional leading minus
        if self._source[self._pos] == "-":
            self._pos += 1

        has_dot = False
        while self._pos < self._length and self._source[self._pos] not in _DELIMITERS:
            ch = self._source[self._pos]
            if ch == ".":
                if has_dot:
                    # Second dot means this isn't a valid number — treat as symbol
                    self._pos = start
                    return self._read_symbol()
                has_dot = True
                self._pos += 1
            elif ch.isdigit():
                self._pos += 1
            else:
                # Non-numeric character — rewind and read as symbol
                self._pos = start
                return self._read_symbol()

        value = self._source[start : self._pos]
        return Token(TokenType.NUMBER, value, start)

    def _read_symbol(self) -> Token:
        """Read a symbol (any non-delimiter sequence)."""
        start = self._pos
        while self._pos < self._length and self._source[self._pos] not in _DELIMITERS:
            self._pos += 1

        value = self._source[start : self._pos]
        if not value:
            raise _make_error(
                start, f"Unexpected character: {self._source[start]!r}", self._source
            )
        return Token(TokenType.SYMBOL, value, start)


def _make_error(offset: int, message: str, source: str) -> ParseError:
    """Create a ParseError with surrounding context."""
    context_start = max(0, offset - 10)
    context_end = min(len(source), offset + 10)
    context_str = source[context_start:context_end]
    return ParseError(offset=offset, message=message, context_str=context_str)
