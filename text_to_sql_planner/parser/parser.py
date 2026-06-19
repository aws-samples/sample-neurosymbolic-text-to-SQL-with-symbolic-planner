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
    IsNotNullNode,
    LimitExpression,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    OrderByExpression,
    QuantifierNode,
    QueryExpression,
    ResultVariable,
    SortCriterion,
    VariableRefNode,
)
from text_to_sql_planner.types.errors import ParseError


_MAX_NESTING_DEPTH = 50

_COMPARISON_OPS = frozenset({"=", "!=", "<", ">", "<=", ">="})
_ARITHMETIC_OPS = frozenset({"+", "-", "*", "/"})
_LOGICAL_BINARY_OPS = frozenset({"and", "or", "implies"})
_BUILTIN_FUNCTIONS = frozenset(
    {"DATE_SUB", "DATE_ADD", "YEAR", "MONTH", "DAY", "DATEDIFF", "LIKE"}
)
_QUANTIFIER_OPS = frozenset({"forall", "exists"})
_AGGREGATE_FUNCS = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})


@dataclass
class ParserSuccess:
    expression: DRCExpression


@dataclass
class ParserFailure:
    error: ParseError


ParserResult = Union[ParserSuccess, ParserFailure]


@dataclass
class QueryParserSuccess:
    """Successful parse of a top-level query (DRC, optionally wrapped in
    ORDER BY / LIMIT non-relational operators)."""

    query: QueryExpression


@dataclass
class QueryParserFailure:
    error: ParseError


QueryParserResult = Union[QueryParserSuccess, QueryParserFailure]


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

    def parse_query(self) -> QueryExpression:
        """Parse a top-level query expression.

        A query is a core DRC expression optionally wrapped in
        ``order-by`` and/or ``limit`` operators. Both wrappers are
        non-relational and live outside the set comprehension. The
        canonical form is::

            (limit N (order-by ((col dir) ...) (drc (...) ...)))

        but either wrapper may be omitted, and ``limit`` need not be the
        outermost layer (for completeness; the SQL emitter will normalize).
        """
        if self._current().type is not TokenType.LPAREN:
            token = self._current()
            raise ParseError(
                offset=token.offset,
                message=f"Expected '(' at start of query, got {token.type.name}",
            )

        # Peek the operator keyword without consuming.
        if self._pos + 1 >= len(self._tokens):
            raise ParseError(
                offset=self._current().offset,
                message="Unexpected end of input after '('",
            )
        head = self._tokens[self._pos + 1]
        if head.type is not TokenType.SYMBOL:
            raise ParseError(
                offset=head.offset,
                message=f"Expected operator symbol, got {head.type.name}",
            )

        if head.value == "drc":
            return self.parse_top()
        if head.value == "limit":
            return self._parse_limit()
        if head.value == "order-by":
            return self._parse_order_by()

        raise ParseError(
            offset=head.offset,
            message=(
                f"Expected 'drc', 'order-by', or 'limit' at top level, "
                f"got '{head.value}'"
            ),
        )

    # ------------------------------------------------------------------
    # Order-by / Limit parsing
    # ------------------------------------------------------------------

    def _parse_limit(self) -> LimitExpression:
        """Parse (limit N <inner-query>)."""
        self._expect(TokenType.LPAREN, "at start of limit")
        self._enter()
        head = self._expect(TokenType.SYMBOL, "expected 'limit' keyword")
        if head.value != "limit":
            raise ParseError(
                offset=head.offset,
                message=f"Expected 'limit' keyword, got '{head.value}'",
            )

        n_token = self._expect(TokenType.NUMBER, "expected positive integer N for limit")
        try:
            n = int(n_token.value)
        except ValueError:
            raise ParseError(
                offset=n_token.offset,
                message=f"limit N must be an integer, got '{n_token.value}'",
            )
        if n <= 0:
            raise ParseError(
                offset=n_token.offset,
                message=f"limit N must be positive, got {n}",
            )

        inner = self.parse_query()

        self._expect(TokenType.RPAREN, "at end of limit")
        self._leave()
        return LimitExpression(n=n, inner=inner)  # type: ignore[arg-type]

    def _parse_order_by(self) -> OrderByExpression:
        """Parse (order-by ((col dir) ...) <inner-query>) where each
        criterion is either a bare column ``col`` or an aggregate form
        ``(AGG col)``, followed by direction ``asc`` or ``desc``.
        """
        self._expect(TokenType.LPAREN, "at start of order-by")
        self._enter()
        head = self._expect(TokenType.SYMBOL, "expected 'order-by' keyword")
        if head.value != "order-by":
            raise ParseError(
                offset=head.offset,
                message=f"Expected 'order-by' keyword, got '{head.value}'",
            )

        # Parse criteria list: ((key1 dir1) (key2 dir2) ...)
        self._expect(TokenType.LPAREN, "before order-by criteria list")
        criteria: list[SortCriterion] = []
        while self._current().type is not TokenType.RPAREN:
            criteria.append(self._parse_sort_criterion())
        self._expect(TokenType.RPAREN, "after order-by criteria list")
        if not criteria:
            token = self._current()
            raise ParseError(
                offset=token.offset,
                message="order-by requires at least one sort criterion",
            )

        inner = self.parse_query()

        self._expect(TokenType.RPAREN, "at end of order-by")
        self._leave()
        return OrderByExpression(criteria=criteria, inner=inner)  # type: ignore[arg-type]

    def _parse_sort_criterion(self) -> SortCriterion:
        """Parse a single ``(key dir)`` sort criterion.

        ``key`` is either a bare column symbol or an aggregate form
        ``(AGG col)``. ``dir`` is ``asc`` or ``desc``.
        """
        self._expect(TokenType.LPAREN, "at start of sort criterion")
        self._enter()

        # Key may be a bare symbol or an aggregate sub-form.
        column = ""
        aggregate: Union[AggregateFunction, None] = None  # type: ignore[assignment]
        if self._current().type is TokenType.LPAREN:
            # Could be (AGG col) or a complex arithmetic expression.
            # Peek the first symbol after '(' to determine.
            self._expect(TokenType.LPAREN, "before aggregate in sort key")
            self._enter()
            agg_tok = self._expect(TokenType.SYMBOL, "expected aggregate function name")
            if agg_tok.value in _AGGREGATE_FUNCS:
                col_tok = self._expect(TokenType.SYMBOL, "expected column name in aggregate sort key")
                self._expect(TokenType.RPAREN, "after aggregate in sort key")
                self._leave()
                aggregate = agg_tok.value  # type: ignore[assignment]
                column = col_tok.value
            elif agg_tok.value in _ARITHMETIC_OPS:
                # Arithmetic expression as sort key — skip it by consuming
                # until the matching closing paren. Use the expression as
                # a synthetic column name "expr".
                depth = 1
                while depth > 0 and not self._at_end():
                    t = self._advance()
                    if t.type is TokenType.LPAREN:
                        depth += 1
                    elif t.type is TokenType.RPAREN:
                        depth -= 1
                self._leave()
                column = "expr"
            else:
                raise ParseError(
                    offset=agg_tok.offset,
                    message=f"Unknown aggregate function in order-by: '{agg_tok.value}'",
                )
        elif self._current().type is TokenType.SYMBOL:
            tok = self._advance()
            column = tok.value
        else:
            token = self._current()
            raise ParseError(
                offset=token.offset,
                message=f"Expected sort key (column or aggregate), got {token.type.name}",
            )

        dir_tok = self._expect(TokenType.SYMBOL, "expected sort direction 'asc' or 'desc'")
        direction = dir_tok.value.lower()
        if direction not in ("asc", "desc"):
            raise ParseError(
                offset=dir_tok.offset,
                message=f"Invalid sort direction '{dir_tok.value}'; expected 'asc' or 'desc'",
            )

        self._expect(TokenType.RPAREN, "at end of sort criterion")
        self._leave()
        return SortCriterion(column=column, direction=direction, aggregate=aggregate)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Result variables
    # ------------------------------------------------------------------

    def _parse_result_variables(self) -> list[ResultVariable]:
        variables: list[ResultVariable] = []
        while self._current().type is not TokenType.RPAREN:
            if self._current().type is TokenType.LPAREN:
                # Could be:
                # - Aggregate: (COUNT col)
                # - Arithmetic aggregate: (/ (COUNT x) (COUNT y))
                variables.append(self._parse_aggregate_or_arithmetic_variable())
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

    def _parse_aggregate_or_arithmetic_variable(self) -> ResultVariable:
        """Parse either ``(AGG col)`` or ``(op (AGG col) (AGG col))``
        or ``(COUNT_IF condition col)``."""
        from text_to_sql_planner.types.drc import ArithmeticAggregateVariable, ConditionalAggregateVariable

        self._expect(TokenType.LPAREN, "before aggregate/arithmetic")
        self._enter()

        head = self._current()
        if head.type is not TokenType.SYMBOL:
            raise ParseError(
                offset=head.offset,
                message=f"Expected function name or operator, got {head.type.name}",
            )

        # Arithmetic operator in result-variable position:
        # (/ (COUNT x) (COUNT y))
        if head.value in ("+", "-", "*", "/"):
            op_token = self._advance()
            left = self._parse_inner_aggregate()
            right = self._parse_inner_aggregate()
            self._expect(TokenType.RPAREN, "after arithmetic aggregate")
            self._leave()
            return ArithmeticAggregateVariable(
                operator=op_token.value,  # type: ignore[arg-type]
                left=left,
                right=right,
            )

        # Conditional aggregate: (COUNT_IF condition col) or (SUM_IF ...)
        if head.value.endswith("_IF") and head.value[:-3] in _AGGREGATE_FUNCS:
            return self._parse_conditional_aggregate_inner(head)

        # Plain aggregate: (COUNT col)
        if head.value not in _AGGREGATE_FUNCS:
            raise ParseError(
                offset=head.offset,
                message=f"Unknown aggregate function: '{head.value}'",
            )
        func_token = self._advance()
        col_token = self._expect(TokenType.SYMBOL, "expected column name in aggregate")
        self._expect(TokenType.RPAREN, "after aggregate variable")
        self._leave()
        return AggregateVariable(
            function=func_token.value,  # type: ignore[arg-type]
            column=col_token.value,
        )

    def _parse_inner_aggregate(self) -> ResultVariable:
        """Parse a parenthesised ``(AGG col)`` or ``(AGG_IF cond col)``
        or nested arithmetic ``(/ (AGG col) (AGG col))``
        inside an arithmetic aggregate.

        Also accepts a bare NUMBER token (not parenthesised) for scalar
        literal operands like ``100`` in ``(* (/ ...) 100)``, and a bare
        SYMBOL token for column variable references like ``cons2013`` in
        ``(- cons2013 cons2012)``.
        """
        # Allow bare number literal without parentheses
        if self._current().type is TokenType.NUMBER:
            tok = self._advance()
            from text_to_sql_planner.types.drc import ScalarLiteralVariable
            if "." in tok.value:
                return ScalarLiteralVariable(value=float(tok.value))
            return ScalarLiteralVariable(value=int(tok.value))

        # Allow bare symbol (column variable reference) without parentheses
        if self._current().type is TokenType.SYMBOL and self._current().value not in _AGGREGATE_FUNCS and not self._current().value.endswith("_IF") and self._current().value not in ("+", "-", "*", "/"):
            tok = self._advance()
            return ColumnVariable(name=tok.value)

        self._expect(TokenType.LPAREN, "before inner aggregate")
        self._enter()
        head = self._current()
        if head.type is not TokenType.SYMBOL:
            raise ParseError(
                offset=head.offset,
                message=f"Expected aggregate function name, got {head.type.name}",
            )
        # Conditional aggregate: (COUNT_IF condition col) / (SUM_IF ...)
        # Routed before the _AGGREGATE_FUNCS membership check so the
        # _IF suffix is recognised inside arithmetic aggregates.
        if head.value.endswith("_IF") and head.value[:-3] in _AGGREGATE_FUNCS:
            return self._parse_conditional_aggregate_inner(head)
        # Nested arithmetic: (/ (COUNT x) (COUNT y)) inside outer arithmetic
        if head.value in ("+", "-", "*", "/"):
            from text_to_sql_planner.types.drc import ArithmeticAggregateVariable
            op_token = self._advance()
            left = self._parse_inner_aggregate()
            right = self._parse_inner_aggregate()
            self._expect(TokenType.RPAREN, "after nested arithmetic aggregate")
            self._leave()
            return ArithmeticAggregateVariable(
                operator=op_token.value,
                left=left,
                right=right,
            )
        func_token = self._advance()
        if func_token.value not in _AGGREGATE_FUNCS:
            # Not an aggregate — treat as a function call (e.g. YEAR, DATEDIFF).
            # Consume all arguments until closing paren, represent as ColumnVariable.
            depth = 1
            while depth > 0 and not self._at_end():
                t = self._advance()
                if t.type is TokenType.LPAREN:
                    depth += 1
                elif t.type is TokenType.RPAREN:
                    depth -= 1
            self._leave()
            return ColumnVariable(name=func_token.value)
        col_token = self._expect(TokenType.SYMBOL, "expected column name")
        self._expect(TokenType.RPAREN, "after inner aggregate")
        self._leave()
        return AggregateVariable(
            function=func_token.value,  # type: ignore[arg-type]
            column=col_token.value,
        )

    def _parse_conditional_aggregate_inner(
        self, head: Token
    ) -> "ConditionalAggregateVariable":
        """Parse the body of a ``(COUNT_IF cond col)`` / ``(SUM_IF cond col)`` form.

        Caller invariants:

        * The current parser position is the head ``SYMBOL`` token.
        * The enclosing ``(`` has already been consumed and ``_enter()``
          called.
        * ``head.value[:-3]`` is in ``_AGGREGATE_FUNCS``.

        The helper consumes the head, the condition, the column ``SYMBOL``,
        and the closing ``)``, then calls ``_leave()`` and returns the
        :class:`ConditionalAggregateVariable` node. ``ParseError`` is
        raised through the existing ``_expect`` machinery on malformed
        input so error messages and offsets stay identical to the
        pre-extraction behaviour.
        """
        from text_to_sql_planner.types.drc import ConditionalAggregateVariable

        func_name = head.value[:-3]  # Strip _IF suffix
        self._advance()
        # Parse the condition (a full DRC condition expression).
        condition = self._parse_condition()
        # Parse the column name.
        col_token = self._expect(
            TokenType.SYMBOL, "expected column name in conditional aggregate"
        )
        self._expect(TokenType.RPAREN, "after conditional aggregate")
        self._leave()
        return ConditionalAggregateVariable(
            function=func_name,  # type: ignore[arg-type]
            column=col_token.value,
            condition=condition,
        )

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
        elif op == "is-not-null":
            col_token = self._expect(
                TokenType.SYMBOL, "expected column name in is-not-null"
            )
            result = IsNotNullNode(column=col_token.value)
        elif op in _BUILTIN_FUNCTIONS or op == "CURRENT_DATE":
            result = self._parse_function_call(op)
        elif op.upper() == "LIKE":
            # The LLM sometimes emits lowercase ``like``; the SQL→DRC
            # translator emits uppercase ``LIKE``. Accept both and
            # normalise to uppercase so the SMT converter sees one
            # consistent function head.
            result = self._parse_function_call("LIKE")
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

    def _parse_logical_binary(self, op: str) -> DRCCondition:
        """Parse n-ary ``(and ...)``, ``(or ...)``, or ``(implies ...)``.

        The DRC AST stores logical connectives as binary nodes
        (``LogicalConnectiveNode`` with ``left`` and ``right`` fields),
        so an n-ary input is left-folded into a chain of binary nodes:

            (and X1 X2 X3)  →  (and (and X1 X2) X3)

        Degenerate cases:

        * ``(op X)`` — single operand. ``and`` and ``or`` of one
          operand are equivalent to the operand itself, so we return
          ``X`` directly. This also accommodates a common LLM mistake
          of wrapping a single conjunct in ``(and ...)``.
        * ``(op)`` — zero operands. Rejected as a ``ParseError`` (no
          sensible identity for ``implies``, and an empty ``and``/``or``
          is suspicious enough to flag).

        ``implies`` requires exactly two operands; n-ary ``implies``
        has no standard meaning.
        """
        operands: list[DRCCondition] = []
        while self._current().type is not TokenType.RPAREN:
            operands.append(self._parse_condition())

        if not operands:
            raise ParseError(
                offset=self._current().offset,
                message=f"'{op}' requires at least one operand",
            )

        if op == "implies":
            if len(operands) != 2:
                raise ParseError(
                    offset=self._current().offset,
                    message=(
                        f"'implies' requires exactly 2 operands, got {len(operands)}"
                    ),
                )
            return LogicalConnectiveNode(
                operator=op,  # type: ignore[arg-type]
                left=operands[0],
                right=operands[1],
            )

        # ``and`` / ``or``: left-fold into a binary chain.
        if len(operands) == 1:
            return operands[0]

        result: DRCCondition = operands[0]
        for next_operand in operands[1:]:
            result = LogicalConnectiveNode(
                operator=op,  # type: ignore[arg-type]
                left=result,
                right=next_operand,
            )
        return result

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


def parse_query(source: str) -> QueryParserResult:
    """Parse a top-level query (DRC, optionally wrapped in order-by/limit).

    Examples of accepted forms::

        (drc (...) ...)
        (order-by ((salary desc)) (drc (...) ...))
        (limit 5 (order-by ((salary desc)) (drc (...) ...)))

    Returns :class:`QueryParserSuccess` or :class:`QueryParserFailure`.
    """
    try:
        tokens = Lexer.tokenize(source)
        parser = _Parser(tokens)
        query = parser.parse_query()
        if not parser._at_end():
            token = parser._current()
            raise ParseError(
                offset=token.offset,
                message=f"Unexpected trailing token: {token.type.name}",
            )
        return QueryParserSuccess(query=query)
    except ParseError as e:
        return QueryParserFailure(error=e)


def parse_condition(source: str):
    """Parse a DRC condition S-expression string into a condition AST node.

    Returns the condition node on success, None on any parse error.
    """
    try:
        tokens = Lexer.tokenize(source)
        parser = _Parser(tokens)
        condition = parser._parse_condition()
        return condition
    except (ParseError, IndexError):
        return None
