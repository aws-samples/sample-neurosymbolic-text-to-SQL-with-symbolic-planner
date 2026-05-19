"""Unit tests for the Lisp S-expression lexer."""

import pytest

from text_to_sql_planner.parser.lexer import Lexer, Token, TokenType
from text_to_sql_planner.types.errors import ParseError


class TestBasicTokenization:
    """Test basic token production."""

    def test_empty_input(self):
        tokens = Lexer.tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type is TokenType.EOF

    def test_whitespace_only(self):
        tokens = Lexer.tokenize("   \t\n  ")
        assert len(tokens) == 1
        assert tokens[0].type is TokenType.EOF

    def test_single_lparen(self):
        tokens = Lexer.tokenize("(")
        assert tokens[0] == Token(TokenType.LPAREN, "(", 0)

    def test_single_rparen(self):
        tokens = Lexer.tokenize(")")
        assert tokens[0] == Token(TokenType.RPAREN, ")", 0)

    def test_symbol(self):
        tokens = Lexer.tokenize("forall")
        assert tokens[0] == Token(TokenType.SYMBOL, "forall", 0)

    def test_integer(self):
        tokens = Lexer.tokenize("42")
        assert tokens[0] == Token(TokenType.NUMBER, "42", 0)

    def test_negative_integer(self):
        tokens = Lexer.tokenize("-7")
        assert tokens[0] == Token(TokenType.NUMBER, "-7", 0)

    def test_float(self):
        tokens = Lexer.tokenize("3.14")
        assert tokens[0] == Token(TokenType.NUMBER, "3.14", 0)

    def test_negative_float(self):
        tokens = Lexer.tokenize("-0.5")
        assert tokens[0] == Token(TokenType.NUMBER, "-0.5", 0)

    def test_string_literal(self):
        tokens = Lexer.tokenize('"hello"')
        assert tokens[0] == Token(TokenType.STRING, "hello", 0)

    def test_string_with_escape_quote(self):
        tokens = Lexer.tokenize(r'"say \"hi\""')
        assert tokens[0].type is TokenType.STRING
        assert tokens[0].value == 'say "hi"'

    def test_string_with_escape_backslash(self):
        tokens = Lexer.tokenize(r'"path\\to"')
        assert tokens[0].type is TokenType.STRING
        assert tokens[0].value == "path\\to"


class TestFullExpressions:
    """Test tokenization of complete expressions."""

    def test_forall_expression(self):
        source = "(forall (x y) (in (x y) Students))"
        tokens = Lexer.tokenize(source)
        types = [t.type for t in tokens]
        expected_types = [
            TokenType.LPAREN,
            TokenType.SYMBOL,   # forall
            TokenType.LPAREN,
            TokenType.SYMBOL,   # x
            TokenType.SYMBOL,   # y
            TokenType.RPAREN,
            TokenType.LPAREN,
            TokenType.SYMBOL,   # in
            TokenType.LPAREN,
            TokenType.SYMBOL,   # x
            TokenType.SYMBOL,   # y
            TokenType.RPAREN,
            TokenType.SYMBOL,   # Students
            TokenType.RPAREN,
            TokenType.RPAREN,
            TokenType.EOF,
        ]
        assert types == expected_types

    def test_forall_expression_values(self):
        source = "(forall (x y) (in (x y) Students))"
        tokens = Lexer.tokenize(source)
        symbols = [t.value for t in tokens if t.type is TokenType.SYMBOL]
        assert symbols == ["forall", "x", "y", "in", "x", "y", "Students"]

    def test_expression_with_numbers(self):
        source = "(> age 21)"
        tokens = Lexer.tokenize(source)
        assert tokens[0].type is TokenType.LPAREN
        assert tokens[1] == Token(TokenType.SYMBOL, ">", 1)
        assert tokens[2] == Token(TokenType.SYMBOL, "age", 3)
        assert tokens[3] == Token(TokenType.NUMBER, "21", 7)
        assert tokens[4].type is TokenType.RPAREN

    def test_expression_with_string(self):
        source = '(= name "Alice")'
        tokens = Lexer.tokenize(source)
        assert tokens[3] == Token(TokenType.STRING, "Alice", 8)

    def test_nested_expression(self):
        source = "(and (> x 5) (< y 10))"
        tokens = Lexer.tokenize(source)
        types = [t.type for t in tokens]
        expected = [
            TokenType.LPAREN,
            TokenType.SYMBOL,   # and
            TokenType.LPAREN,
            TokenType.SYMBOL,   # >
            TokenType.SYMBOL,   # x
            TokenType.NUMBER,   # 5
            TokenType.RPAREN,
            TokenType.LPAREN,
            TokenType.SYMBOL,   # <
            TokenType.SYMBOL,   # y
            TokenType.NUMBER,   # 10
            TokenType.RPAREN,
            TokenType.RPAREN,
            TokenType.EOF,
        ]
        assert types == expected


class TestOffsetTracking:
    """Test that character offsets are correctly tracked."""

    def test_offsets_simple(self):
        tokens = Lexer.tokenize("(a b)")
        assert tokens[0].offset == 0  # (
        assert tokens[1].offset == 1  # a
        assert tokens[2].offset == 3  # b
        assert tokens[3].offset == 4  # )

    def test_offsets_with_whitespace(self):
        tokens = Lexer.tokenize("  (  hello  )")
        assert tokens[0].offset == 2   # (
        assert tokens[1].offset == 5   # hello
        assert tokens[2].offset == 12  # )

    def test_eof_offset(self):
        tokens = Lexer.tokenize("abc")
        eof = tokens[-1]
        assert eof.type is TokenType.EOF
        assert eof.offset == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unterminated_string_raises_parse_error(self):
        with pytest.raises(ParseError):
            Lexer.tokenize('"unterminated')

    def test_unterminated_escape_raises_parse_error(self):
        with pytest.raises(ParseError):
            Lexer.tokenize('"end with backslash\\')

    def test_minus_alone_is_symbol(self):
        """A lone '-' without a following digit is a symbol."""
        tokens = Lexer.tokenize("-")
        assert tokens[0] == Token(TokenType.SYMBOL, "-", 0)

    def test_minus_followed_by_letter_is_symbol(self):
        tokens = Lexer.tokenize("-abc")
        assert tokens[0] == Token(TokenType.SYMBOL, "-abc", 0)

    def test_symbol_with_special_chars(self):
        """Symbols can contain characters like >, <, =, !, etc."""
        tokens = Lexer.tokenize(">=")
        assert tokens[0] == Token(TokenType.SYMBOL, ">=", 0)

    def test_multiple_dots_is_symbol(self):
        """A number-like token with multiple dots falls back to symbol."""
        tokens = Lexer.tokenize("1.2.3")
        assert tokens[0].type is TokenType.SYMBOL
        assert tokens[0].value == "1.2.3"


class TestIteratorProtocol:
    """Test that the lexer works as an iterator."""

    def test_iter_produces_tokens(self):
        lexer = Lexer("(a)")
        tokens = list(lexer)
        assert len(tokens) == 3
        assert tokens[0].type is TokenType.LPAREN
        assert tokens[1].type is TokenType.SYMBOL
        assert tokens[2].type is TokenType.RPAREN

    def test_next_token_includes_eof(self):
        lexer = Lexer("")
        token = lexer.next_token()
        assert token.type is TokenType.EOF
