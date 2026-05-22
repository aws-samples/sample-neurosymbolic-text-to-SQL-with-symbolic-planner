"""Recursive descent parser for Lisp S-expression DRC syntax.

Converts tokenized S-expressions into DRC AST nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from text_to_sql_planner.parser.lexer import Lexer, Token, TokenType
from text_to_sql_planner.types.drc import (
    AggregateFunction,
    AggregateVariable,
    ArithmeticNode,
    ColumnVariable,
    ComparisonNode,
    DRCCondition,
    DRCExpression,
    FunctionCallNode,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    ResultVariable,
    VariableRefNode,
)
from text_to_sql_planner.types.errors import ParseError


_MAX_NESTING_DEPTH = 50

_COMPARISON_OPS = frozenset({"=", "!=", "<", ">", "<=", ">="})
_ARITHMETIC_OPS = frozenset({"+", "-", "*", "/"})
_LOGICAL_BINARY_OPS = frozenset({"and", "or", "implies"})
_BUILTIN_FUNCTIONS = frozenset({"DATE_SUB", "DATE_ADD", "YEAR", "MONTH", "DAY", "DATEDIFF"})
_QUANTIFIER_OPS = frozenset({"forall", "exists"})
_AGGREGATE_FUNCS = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})


@dataclass
class ParserSuccess:
    expression: DRCExpression


@dataclass
class ParserFailure:
    error: ParseError


ParserResult = Union[ParserSuccess, ParserFailure]


class _Parser:
    """Internal recursive descent parser state."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._depth = 0

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _current(self) -> Token:
        return self._tokens[self._pos]

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _expect(self, token_type: TokenType, context: str = "") -> Token:
        token = self._current()
        if token.type is not token_type:
            msg = f"Expected {token_type.name}"
            if context:
                msg += f" {context}"
            msg += f", got {token.type.name}"
            raise ParseError(offset=token.offset, message=msg)
        return self._advance()

    def _at_end(self) -> bool:
        return self._current().type is TokenType.EOF

    # ------------------------------------------------------------------
    # Depth tracking
    # ------------------------------------------------------------------

    def _enter(self) -> None:
        self._depth += 1
        if self._depth > _MAX_NESTING_DEPTH:
            token = self._current()
            raise ParseError(
                offset=token.offset,
                message="max nesting depth exceeded",
            )

    def _leave(self) -> None:
        self._depth -= 1

    # ------------------------------------------------------------------
    # Top-level parse
    # ------------------------------------------------------------------

    def parse_top(self) -> DRCExpression:
        """Parse a full (drc (result_vars...) condition) expression."""
        self._expect(TokenType.LPAREN, "at start of drc expression")
        self._enter()

        sym = self._expect(TokenType.SYMBOL, "expected 'drc' keyword")
        if sym.value != "drc":
            raise ParseError(
                offset=sym.offset,
                message=f"Expected 'drc' keyword, got '{sym.value}'",
            )

        # Parse result variables list
        self._expect(TokenType.LPAREN, "before result variables")
        result_vars = self._parse_result_variables()
        self._expect(TokenType.RPAREN, "after result variables")

        # Parse condition
        condition = self._parse_condition()

        self._expect(TokenType.RPAREN, "at end of drc expression")
        self._leave()

        return DRCExpression(result_variables=result_vars, condition=condition)

    # ------------------------------------------------------------------
    # Result variables
    # ------------------------------------------------------------------

    def _parse_result_variables(self) -> list[ResultVariable]:
        variables: list[ResultVariable] = []
        while self._current().type is not TokenType.RPAREN:
            if self._current().type is TokenType.LPAREN:
                # Aggregate variable: (AGG col)
                variables.append(self._parse_aggregate_variable())
            elif self._current().type is TokenType.SYMBOL:
                token = self._advance()
                variables.append(ColumnVariable(name=token.value))
            else:
                token = self._current()
                raise ParseError(
                    offset=token.offset,
                    message=f"Unexpected token in result variables: {token.type.name}",
                )
        return variables

    def _parse_aggregate_variable(self) -> AggregateVariable:
        self._expect(TokenType.LPAREN, "before aggregate function")
        self._enter()

        func_token = self._expect(TokenType.SYMBOL, "expected aggregate function name")
        if func_token.value not in _AGGREGATE_FUNCS:
            raise ParseError(
                offset=func_token.offset,
                message=f"Unknown aggregate function: '{func_token.value}'",
            )

        col_token = self._expect(TokenType.SYMBOL, "expected column name in aggregate")

        self._expect(TokenType.RPAREN, "after aggregate variable")
        self._leave()

        return AggregateVariable(
            function=func_token.value,  # type: ignore[arg-type]
            column=col_token.value,
        )

    # ------------------------------------------------------------------
    # Condition parsing
    # ------------------------------------------------------------------

    def _parse_condition(self) -> DRCCondition:
        token = self._current()

        if token.type is TokenType.LPAREN:
            return self._parse_compound()
        elif token.type is TokenType.STRING:
            self._advance()
            return LiteralNode(value=token.value, data_type="string")
        elif token.type is TokenType.NUMBER:
            self._advance()
            value: Union[int, float]
            if "." in token.value:
                value = float(token.value)
            else:
                value = int(token.value)
            return LiteralNode(value=value, data_type="number")
        elif token.type is TokenType.SYMBOL:
            self._advance()
            # CURRENT_DATE is a built-in constant (no arguments)
            if token.value == "CURRENT_DATE":
                return FunctionCallNode(function="CURRENT_DATE", arguments=[])
            return VariableRefNode(name=token.value)
        else:
            raise ParseError(
                offset=token.offset,
                message=f"Unexpected token: {token.type.name}",
            )

    def _parse_compound(self) -> DRCCondition:
        """Parse a parenthesized compound expression."""
        self._expect(TokenType.LPAREN)
        self._enter()

        op_token = self._current()
        if op_token.type is not TokenType.SYMBOL:
            raise ParseError(
                offset=op_token.offset,
                message=f"Expected operator symbol, got {op_token.type.name}",
            )

        op = op_token.value
        self._advance()

        result: DRCCondition

        if op in _QUANTIFIER_OPS:
            result = self._parse_quantifier(op, op_token)
        elif op in _LOGICAL_BINARY_OPS:
            result = self._parse_logical_binary(op)
        elif op == "not":
            result = self._parse_not()
        elif op in _COMPARISON_OPS:
            result = self._parse_comparison(op)
        elif op in _ARITHMETIC_OPS:
            result = self._parse_arithmetic(op)
        elif op == "in":
            result = self._parse_membership()
        elif op in _BUILTIN_FUNCTIONS or op == "CURRENT_DATE":
            result = self._parse_function_call(op)
        else:
            raise ParseError(
                offset=op_token.offset,
                message=f"Unknown operator: '{op}'",
            )

        self._expect(TokenType.RPAREN, f"at end of '{op}' expression")
        self._leave()

        return result

    def _parse_quantifier(self, kind: str, op_token: Token) -> QuantifierNode:
        """Parse (forall (vars...) body) or (exists (vars...) body)."""
        # Parse variable list
        self._expect(TokenType.LPAREN, "before quantifier variables")
        variables: list[str] = []
        while self._current().type is not TokenType.RPAREN:
            var_token = self._expect(TokenType.SYMBOL, "in quantifier variable list")
            variables.append(var_token.value)
        self._expect(TokenType.RPAREN, "after quantifier variables")

        # Parse body condition
        body = self._parse_condition()

        return QuantifierNode(
            kind=kind,  # type: ignore[arg-type]
            variables=variables,
            body=body,
        )

    def _parse_logical_binary(self, op: str) -> LogicalConnectiveNode:
        """Parse (and left right), (or left right), (implies left right)."""
        left = self._parse_condition()
        right = self._parse_condition()
        return LogicalConnectiveNode(
            operator=op,  # type: ignore[arg-type]
            left=left,
            right=right,
        )

    def _parse_not(self) -> NotNode:
        """Parse (not operand)."""
        operand = self._parse_condition()
        return NotNode(operand=operand)

    def _parse_comparison(self, op: str) -> ComparisonNode:
        """Parse (= left right), (!= left right), etc."""
        left = self._parse_condition()
        right = self._parse_condition()
        return ComparisonNode(
            operator=op,  # type: ignore[arg-type]
            left=left,
            right=right,
        )

    def _parse_arithmetic(self, op: str) -> ArithmeticNode:
        """Parse (+ left right), (- left right), etc."""
        left = self._parse_condition()
        right = self._parse_condition()
        return ArithmeticNode(
            operator=op,  # type: ignore[arg-type]
            left=left,
            right=right,
        )

    def _parse_membership(self) -> MembershipNode:
        """Parse (in (vars...) RelationName)."""
        self._expect(TokenType.LPAREN, "before membership variables")
        variables: list[str] = []
        while self._current().type is not TokenType.RPAREN:
            var_token = self._expect(TokenType.SYMBOL, "in membership variable list")
            variables.append(var_token.value)
        self._expect(TokenType.RPAREN, "after membership variables")

        relation_token = self._expect(TokenType.SYMBOL, "expected relation name")

        return MembershipNode(variables=variables, relation=relation_token.value)

    def _parse_function_call(self, func_name: str) -> FunctionCallNode:
        """Parse (FUNC_NAME arg1 arg2 ...) — built-in function call."""
        arguments: list = []
        while self._current().type is not TokenType.RPAREN:
            arguments.append(self._parse_condition())
        return FunctionCallNode(function=func_name, arguments=arguments)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def parse(source: str) -> ParserResult:
    """Parse a DRC S-expression string into an AST.

    Returns ParserSuccess on success, ParserFailure on any parse error.
    """
    try:
        tokens = Lexer.tokenize(source)
        parser = _Parser(tokens)
        expression = parser.parse_top()

        # Ensure no trailing tokens
        if not parser._at_end():
            token = parser._current()
            raise ParseError(
                offset=token.offset,
                message=f"Unexpected trailing token: {token.type.name}",
            )

        return ParserSuccess(expression=expression)
    except ParseError as e:
        return ParserFailure(error=e)
