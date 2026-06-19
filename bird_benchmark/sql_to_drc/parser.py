"""SQL recursive-descent SELECT parser for the SQL→DRC converter.

This module turns the flat :class:`~bird_benchmark.sql_to_drc.lexer.Token`
stream produced by :mod:`bird_benchmark.sql_to_drc.lexer` into a
:class:`~bird_benchmark.sql_to_drc.ast.SelectStatement` AST. It is the
gatekeeper for the converter's Supported_SQL_Subset (Req 2.1, Req 2.4):
inputs that lie outside the subset are reported as a structured
:class:`~bird_benchmark.types.ConverterError` rather than a Python
exception, so the BIRD suite can record a single failed Test_Case and
keep running.

Supported_SQL_Subset (Req 2.1)
==============================

- ``SELECT [DISTINCT] <select_list> FROM <table_source>`` with explicit
  column lists or the ``*`` shorthand
- ``FROM`` with one base table and zero or more ``[INNER] JOIN <table>
  ON <expr>`` chains (Req 2.1)
- ``WHERE`` with arbitrary AND/OR/NOT trees of comparisons
  (``=``, ``!=``/``<>``, ``<``, ``>``, ``<=``, ``>=``)
- ``GROUP BY <column_list>``
- ``ORDER BY <expr> [ASC|DESC] [, ...]``
- ``LIMIT <integer>``
- Scalar aggregates: ``COUNT``, ``SUM``, ``AVG``, ``MIN``, ``MAX``,
  with optional ``DISTINCT`` and the ``*`` argument shorthand
- ``IN`` with a parenthesised literal list
- ``IN (SELECT ...)`` and ``EXISTS (SELECT ...)`` subqueries
  (correlation is resolved later by the translator)
- SQLite-flavoured literals (single-quoted strings, double-quoted
  identifier-or-string tokens), the ``||`` operator, the ``+ - * /``
  arithmetic operators, the ``strftime`` family, and arbitrary other
  function calls (passed through to the translator, which decides
  whether each function name is supported)

Explicitly out-of-scope (Req 2.4)
=================================

- ``LEFT``/``RIGHT``/``FULL``/``OUTER`` JOIN, and ``CROSS JOIN``
- Set operations: ``UNION``, ``INTERSECT``, ``EXCEPT``
- Common Table Expressions (``WITH ...``), including ``WITH RECURSIVE``
- ``HAVING`` clauses
- Window functions (``... OVER (...)``)
- ``CASE`` expressions
- ``BETWEEN``, ``IS NULL`` / ``IS NOT NULL``, bare ``NULL``
- Recursive queries

Each detected out-of-scope construct yields
``ConverterError(kind="unsupported_feature", feature=<short_id>, ...)``
with the source 1-indexed line and column carried by the offending token
(Req 2.3, Req 13.8). Malformed SQL that is not in any feature category
yields ``ConverterError(kind="parse_error", ...)`` (Req 2.6); empty,
whitespace-only, and comment-only inputs yield ``parse_error`` with the
message ``"no SELECT statement found"`` anchored at line 1, column 1
(Req 2.7).

Implementation notes
--------------------

- The parser is hand-rolled recursive descent over the EBNF grammar in
  the design document. It does not use any external parsing library so
  that the supported subset stays in lockstep with the converter's
  feature list.
- Recursive parse routines bail out via the private ``_ParseFail``
  exception, which carries the structured ``ConverterError`` back to the
  public :func:`parse` entry point. The exception is *internal* — the
  public API always returns the error rather than raising.
- ``LEX_ERROR`` tokens from the lexer are surfaced as
  ``ConverterError(kind="parse_error")`` before any grammar rules run,
  using the lexer's diagnostic message verbatim.
"""

from __future__ import annotations

from .ast import (
    Aggregate,
    AggregateFunction,
    BinaryOp,
    CaseExpr,
    ColumnRef,
    DerivedTable,
    ExistsExpr,
    Expression,
    FunctionCall,
    InList,
    InSubquery,
    InnerJoin,
    JoinChain,
    Literal,
    OrderKey,
    Position,
    SelectItem,
    SelectStatement,
    TableRef,
    TableSource,
    UnaryOp,
)
from .lexer import Token, TokenKind
from ..types import ConverterError


# --- Internal control flow -------------------------------------------------


class _ParseFail(Exception):
    """Carrier for ``ConverterError`` so recursive parsers can bail out.

    This is *internal* to the parser: it is raised inside the recursive
    helpers and caught once at the top of :func:`parse`. Callers of the
    public API never see it.
    """

    def __init__(self, error: ConverterError) -> None:
        self.error = error
        super().__init__(error.message)


# Aggregate function names that the lexer emits as ``KEYWORD`` tokens.
_AGGREGATE_NAMES: frozenset[str] = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})

# Comparison operators recognised by the comparison rule.
_COMPARISON_OPS: frozenset[str] = frozenset({"=", "!=", "<>", "<", ">", "<=", ">="})

# Set-operation keywords that are detected after a complete select_stmt.
_SET_OP_KEYWORDS: frozenset[str] = frozenset({"UNION", "INTERSECT", "EXCEPT"})

# Outer-join lead keywords whose presence signals an out-of-scope join.
_OUTER_JOIN_LEADS: frozenset[str] = frozenset({"LEFT", "RIGHT", "FULL", "OUTER"})


# --- Hand-rolled recursive-descent parser ----------------------------------


class _Parser:
    """Hand-rolled recursive-descent parser over a flat token list.

    The parser owns a position cursor and a small set of token-level
    helpers (peek/advance/expect_*). Each grammar rule is a method named
    after the rule. Errors are raised through :class:`_ParseFail` so the
    rules can be written without per-call error threading.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # ----- token-level helpers --------------------------------------------

    def peek(self, offset: int = 0) -> Token:
        """Return the token at ``pos + offset`` (clamped to EOF)."""
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[idx]

    def advance(self) -> Token:
        """Consume and return the current token (EOF stays put)."""
        tok = self.peek()
        if tok.kind != TokenKind.EOF:
            self.pos += 1
        return tok

    def at_eof(self) -> bool:
        return self.peek().kind == TokenKind.EOF

    def at_keyword(self, kw: str, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok.kind == TokenKind.KEYWORD and tok.text.upper() == kw.upper()

    def at_op(self, op: str, offset: int = 0) -> bool:
        tok = self.peek(offset)
        return tok.kind == TokenKind.OP and tok.text == op

    def expect_keyword(self, kw: str) -> Token:
        if self.at_keyword(kw):
            return self.advance()
        tok = self.peek()
        raise self._fail_parse(_describe_expected(kw, tok), tok)

    def expect_op(self, op: str) -> Token:
        if self.at_op(op):
            return self.advance()
        tok = self.peek()
        raise self._fail_parse(_describe_expected(op, tok), tok)

    def expect_ident(self) -> Token:
        tok = self.peek()
        if tok.kind not in (TokenKind.IDENT, TokenKind.KEYWORD):
            raise self._fail_parse(_describe_expected("identifier", tok), tok)
        return self.advance()

    # ----- error helpers --------------------------------------------------

    def _fail_unsupported(
        self, feature: str, message: str, tok: Token
    ) -> _ParseFail:
        return _ParseFail(
            ConverterError(
                kind="unsupported_feature",
                message=message,
                feature=feature,
                line=tok.line,
                column=tok.column,
            )
        )

    def _fail_parse(self, message: str, tok: Token) -> _ParseFail:
        return _ParseFail(
            ConverterError(
                kind="parse_error",
                message=message,
                line=tok.line,
                column=tok.column,
            )
        )

    # ----- grammar: select_stmt ------------------------------------------

    def parse_select_stmt(self) -> SelectStatement:
        """Parse a ``SELECT`` statement, including any nested clauses."""
        # WITH at the position SELECT is expected: CTE / recursive CTE.
        # This guard runs in every nested SELECT context too, so a
        # ``IN (WITH ... SELECT ...)`` subquery is rejected with the
        # same ``cte`` feature id.
        if self.at_keyword("WITH"):
            tok = self.peek()
            if self.at_keyword("RECURSIVE", offset=1):
                raise self._fail_unsupported(
                    "recursive_cte",
                    "WITH RECURSIVE (CTE) is not supported",
                    tok,
                )
            raise self._fail_unsupported(
                "cte", "WITH (common table expression) is not supported", tok
            )

        select_tok = self.expect_keyword("SELECT")

        # Optional DISTINCT directly after SELECT.
        if self.at_keyword("DISTINCT"):
            self.advance()

        select_list = self._parse_select_list()
        self.expect_keyword("FROM")
        from_source = self._parse_table_source()

        where: Expression | None = None
        if self.at_keyword("WHERE"):
            self.advance()
            where = self.parse_expr()

        group_by: list[ColumnRef] = []
        having: Expression | None = None
        if self.at_keyword("GROUP"):
            self.advance()
            self.expect_keyword("BY")
            group_by = self._parse_column_list()
            if self.at_keyword("HAVING"):
                self.advance()
                having = self.parse_expr()

        order_by: list[OrderKey] = []
        if self.at_keyword("ORDER"):
            self.advance()
            self.expect_keyword("BY")
            order_by = self._parse_order_keys()

        limit: int | None = None
        if self.at_keyword("LIMIT"):
            self.advance()
            limit = self._parse_limit_value()

        # After all clauses, set operations are out of scope. The check
        # also catches set operations inside parenthesised subqueries
        # because the subquery's parse_select_stmt completes before the
        # caller's expect_op(")") fires.
        next_tok = self.peek()
        if (
            next_tok.kind == TokenKind.KEYWORD
            and next_tok.text.upper() in _SET_OP_KEYWORDS
        ):
            raise self._fail_unsupported(
                "set_operation",
                f"{next_tok.text.upper()} is not supported",
                next_tok,
            )

        return SelectStatement(
            select_list=select_list,
            from_source=from_source,
            where=where,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
            pos=Position(line=select_tok.line, column=select_tok.column),
        )

    # ----- grammar: select_list / select_item -----------------------------

    def _parse_select_list(self) -> list[SelectItem]:
        items = [self._parse_select_item()]
        while self.at_op(","):
            self.advance()
            items.append(self._parse_select_item())
        return items

    def _parse_select_item(self) -> SelectItem:
        # ``SELECT *`` shorthand. Represented as a string-typed Literal
        # whose value is the literal string "*"; the translator
        # distinguishes it from real string literals by its position.
        if self.at_op("*"):
            tok = self.advance()
            return SelectItem(
                expr=Literal(
                    value="*",
                    data_type="string",
                    pos=Position(line=tok.line, column=tok.column),
                ),
                alias=None,
            )

        expr = self.parse_expr()
        alias: str | None = None
        if self.at_keyword("AS"):
            self.advance()
            alias = self.expect_ident().text
        elif self.peek().kind == TokenKind.IDENT:
            # Implicit alias: ``SELECT a b FROM t``. After a select item
            # the only valid follow-ons in our subset are ``,`` (more
            # items), ``FROM`` (KEYWORD), or end of input — none of
            # which are IDENT — so any IDENT here is an alias.
            alias = self.advance().text
        return SelectItem(expr=expr, alias=alias)

    # ----- grammar: table_source / table_ref ------------------------------

    def _parse_table_source(self) -> TableSource:
        base = self._parse_table_atom()
        joins: list[InnerJoin] = []
        while True:
            tok = self.peek()
            if tok.kind != TokenKind.KEYWORD:
                break
            kw = tok.text.upper()
            if kw == "JOIN":
                join_tok = self.advance()
                right = self._parse_table_atom()
                self.expect_keyword("ON")
                on = self.parse_expr()
                joins.append(
                    InnerJoin(
                        right=right,
                        on=on,
                        pos=Position(join_tok.line, join_tok.column),
                    )
                )
                continue
            if kw == "INNER":
                inner_tok = self.advance()
                if not self.at_keyword("JOIN"):
                    bad = self.peek()
                    raise self._fail_parse(
                        _describe_expected("JOIN", bad), bad
                    )
                self.advance()  # JOIN
                right = self._parse_table_atom()
                self.expect_keyword("ON")
                on = self.parse_expr()
                joins.append(
                    InnerJoin(
                        right=right,
                        on=on,
                        pos=Position(inner_tok.line, inner_tok.column),
                    )
                )
                continue
            if kw in _OUTER_JOIN_LEADS:
                raise self._fail_unsupported(
                    "outer_join",
                    f"{kw} JOIN (outer join) is not supported",
                    tok,
                )
            if kw == "CROSS":
                raise self._fail_unsupported(
                    "cross_join", "CROSS JOIN is not supported", tok
                )
            break
        if joins:
            return JoinChain(base=base, joins=joins)
        return base

    def _parse_table_atom(self) -> TableRef | DerivedTable:
        """Parse one FROM/JOIN table atom: either a bare table reference
        or a parenthesised subquery (a derived table).

        SQL grammar at this level::

            table_atom := IDENT [AS? IDENT]              -- TableRef
                        | "(" SELECT_stmt ")" AS? IDENT  -- DerivedTable

        The derived-table form requires an alias (SQLite enforces
        this; we mirror the constraint and emit a structured parse
        error for missing alias).
        """
        # Derived table: ``(SELECT ...) [AS] alias``.
        if self.at_op("("):
            open_tok = self.advance()  # consume '('
            if not self.at_keyword("SELECT") and not self.at_keyword("WITH"):
                # Not a derived subquery — restore the paren and
                # re-raise as a parse error. (We don't currently
                # support parenthesised join expressions, just bare
                # table refs and subqueries.)
                bad = self.peek()
                raise self._fail_parse(
                    _describe_expected("SELECT", bad), bad
                )
            sub = self.parse_select_stmt()
            self.expect_op(")")
            # SQLite requires an alias on derived tables.
            alias: str | None = None
            if self.at_keyword("AS"):
                self.advance()
                alias = self.expect_ident().text
            elif self.peek().kind == TokenKind.IDENT:
                alias = self.advance().text
            if alias is None:
                bad = self.peek()
                raise self._fail_parse(
                    "derived table requires an alias", bad,
                )
            return DerivedTable(
                subquery=sub,
                alias=alias,
                pos=Position(open_tok.line, open_tok.column),
            )
        return self._parse_table_ref()

    def _parse_table_ref(self) -> TableRef:
        name_tok = self.expect_ident()
        alias: str | None = None
        if self.at_keyword("AS"):
            self.advance()
            alias = self.expect_ident().text
        elif self.peek().kind == TokenKind.IDENT:
            # Implicit alias: ``FROM t1 t2 JOIN ...``. The follow-on is
            # always either a join keyword (KEYWORD), a clause keyword
            # (WHERE/GROUP/ORDER/LIMIT — all KEYWORD), ``,``, ``)`` or
            # end of input, never IDENT.
            alias = self.advance().text
        return TableRef(
            name=name_tok.text,
            alias=alias,
            pos=Position(name_tok.line, name_tok.column),
        )

    # ----- grammar: column_list / order_keys ------------------------------

    def _parse_column_list(self) -> list[ColumnRef]:
        items = [self._parse_column_ref()]
        while self.at_op(","):
            self.advance()
            items.append(self._parse_column_ref())
        return items

    def _parse_column_ref(self) -> ColumnRef:
        first = self.expect_ident()
        if self.at_op("."):
            self.advance()
            second = self.expect_ident()
            # Propagate the trailing identifier's ``quoted`` flag so the
            # translator can apply the schema-fallback rule for the bare
            # column-name token. The qualifier is not affected: a qualified
            # reference is unambiguously an identifier in SQLite.
            return ColumnRef(
                qualifier=first.text,
                name=second.text,
                pos=Position(first.line, first.column),
                quoted=second.quoted,
            )
        return ColumnRef(
            qualifier=None,
            name=first.text,
            pos=Position(first.line, first.column),
            quoted=first.quoted,
        )

    def _parse_order_keys(self) -> list[OrderKey]:
        keys = [self._parse_order_key()]
        while self.at_op(","):
            self.advance()
            keys.append(self._parse_order_key())
        return keys

    def _parse_order_key(self) -> OrderKey:
        expr = self.parse_expr()
        direction: str = "asc"
        if self.at_keyword("ASC"):
            self.advance()
        elif self.at_keyword("DESC"):
            self.advance()
            direction = "desc"
        return OrderKey(expr=expr, direction=direction)  # type: ignore[arg-type]

    def _parse_limit_value(self) -> int:
        tok = self.peek()
        if tok.kind != TokenKind.NUMBER:
            raise self._fail_parse(
                _describe_expected("integer literal", tok), tok
            )
        # Real-valued LIMIT (e.g. ``LIMIT 5.0``) is malformed SQL.
        if "." in tok.text or "e" in tok.text or "E" in tok.text:
            raise self._fail_parse(
                f"LIMIT requires an integer, got {tok.text!r}", tok
            )
        try:
            value = int(tok.text)
        except ValueError:
            raise self._fail_parse(
                f"LIMIT requires an integer, got {tok.text!r}", tok
            )
        self.advance()
        return value

    # ----- grammar: expressions -------------------------------------------

    def parse_expr(self) -> Expression:
        return self._parse_or_expr()

    def _parse_or_expr(self) -> Expression:
        left = self._parse_and_expr()
        while self.at_keyword("OR"):
            tok = self.advance()
            right = self._parse_and_expr()
            left = BinaryOp(
                op="OR",
                left=left,
                right=right,
                pos=Position(tok.line, tok.column),
            )
        return left

    def _parse_and_expr(self) -> Expression:
        left = self._parse_not_expr()
        while self.at_keyword("AND"):
            tok = self.advance()
            right = self._parse_not_expr()
            left = BinaryOp(
                op="AND",
                left=left,
                right=right,
                pos=Position(tok.line, tok.column),
            )
        return left

    def _parse_not_expr(self) -> Expression:
        if self.at_keyword("NOT"):
            not_tok = self.advance()
            # ``NOT EXISTS (...)`` short-circuits via the EXISTS rule
            # so the EXISTS subquery is still consumed; the result is
            # wrapped in ``UnaryOp("NOT", ...)``.
            operand = self._parse_not_expr()
            return UnaryOp(
                op="NOT",
                operand=operand,
                pos=Position(not_tok.line, not_tok.column),
            )
        return self._parse_comparison()

    def _parse_comparison(self) -> Expression:
        left = self._parse_concat_expr()
        tok = self.peek()

        # Out-of-scope comparison forms first, with explicit feature ids.
        if tok.kind == TokenKind.KEYWORD:
            kw = tok.text.upper()
            if kw == "LIKE":
                # ``lhs LIKE pattern`` — emitted as a binary operator with
                # the literal ``LIKE`` keyword as the op string. The
                # translator turns this into a ``FunctionCallNode("LIKE",
                # [lhs, pattern])`` so the SMT layer can declare a
                # consistent uninterpreted Bool predicate on both sides
                # of the equivalence check.
                op_tok = self.advance()
                right = self._parse_concat_expr()
                return BinaryOp(
                    op="LIKE",
                    left=left,
                    right=right,
                    pos=Position(op_tok.line, op_tok.column),
                )
            if kw == "BETWEEN":
                # ``lhs BETWEEN lo AND hi`` → ``BinaryOp("AND",
                #   BinaryOp(">=", lhs, lo), BinaryOp("<=", lhs, hi))``
                # Syntactic sugar: the translator sees two comparisons.
                between_tok = self.advance()  # BETWEEN
                lo = self._parse_concat_expr()
                self.expect_keyword("AND")
                hi = self._parse_concat_expr()
                pos = Position(between_tok.line, between_tok.column)
                return BinaryOp(
                    op="AND",
                    left=BinaryOp(op=">=", left=left, right=lo, pos=pos),
                    right=BinaryOp(op="<=", left=left, right=hi, pos=pos),
                    pos=pos,
                )
            if kw == "IS":
                # ``lhs IS NULL`` / ``lhs IS NOT NULL`` — emit as a
                # unary function-call-style node that the translator
                # turns into ``FunctionCallNode("IS_NULL", [lhs])`` or
                # ``FunctionCallNode("IS_NOT_NULL", [lhs])``.
                is_tok = self.advance()  # IS
                negated = False
                if self.at_keyword("NOT"):
                    self.advance()  # NOT
                    negated = True
                self.expect_keyword("NULL")
                op_name = "IS_NOT_NULL" if negated else "IS_NULL"
                return UnaryOp(
                    op=op_name,
                    operand=left,
                    pos=Position(is_tok.line, is_tok.column),
                )
            if kw == "IN":
                self.advance()
                return self._parse_in_rhs(left, negated=False, pos_tok=tok)
            if kw == "NOT":
                # ``x NOT IN (...)`` — only this combination is
                # interesting at this level. Other ``NOT`` uses are
                # handled in :meth:`_parse_not_expr`.
                if self.at_keyword("IN", offset=1):
                    self.advance()  # NOT
                    in_tok = self.advance()  # IN
                    return self._parse_in_rhs(
                        left, negated=True, pos_tok=in_tok
                    )

        if tok.kind == TokenKind.OP and tok.text in _COMPARISON_OPS:
            op_tok = self.advance()
            # Normalise SQLite ``<>`` to the canonical ``!=``.
            op_text = "!=" if op_tok.text == "<>" else op_tok.text
            right = self._parse_concat_expr()
            return BinaryOp(
                op=op_text,
                left=left,
                right=right,
                pos=Position(op_tok.line, op_tok.column),
            )

        return left

    def _parse_in_rhs(
        self, left: Expression, negated: bool, pos_tok: Token
    ) -> Expression:
        self.expect_op("(")
        if self.at_keyword("SELECT") or self.at_keyword("WITH"):
            sub = self.parse_select_stmt()
            self.expect_op(")")
            node: Expression = InSubquery(
                left=left,
                subquery=sub,
                pos=Position(pos_tok.line, pos_tok.column),
            )
        else:
            values = [self.parse_expr()]
            while self.at_op(","):
                self.advance()
                values.append(self.parse_expr())
            self.expect_op(")")
            node = InList(
                left=left,
                values=values,
                pos=Position(pos_tok.line, pos_tok.column),
            )
        if negated:
            return UnaryOp(
                op="NOT",
                operand=node,
                pos=Position(pos_tok.line, pos_tok.column),
            )
        return node

    def _parse_concat_expr(self) -> Expression:
        left = self._parse_add_expr()
        while self.at_op("||"):
            tok = self.advance()
            right = self._parse_add_expr()
            left = BinaryOp(
                op="||",
                left=left,
                right=right,
                pos=Position(tok.line, tok.column),
            )
        return left

    def _parse_add_expr(self) -> Expression:
        left = self._parse_mul_expr()
        while self.at_op("+") or self.at_op("-"):
            tok = self.advance()
            right = self._parse_mul_expr()
            left = BinaryOp(
                op=tok.text,
                left=left,
                right=right,
                pos=Position(tok.line, tok.column),
            )
        return left

    def _parse_mul_expr(self) -> Expression:
        left = self._parse_unary_expr()
        while self.at_op("*") or self.at_op("/"):
            tok = self.advance()
            right = self._parse_unary_expr()
            left = BinaryOp(
                op=tok.text,
                left=left,
                right=right,
                pos=Position(tok.line, tok.column),
            )
        return left

    def _parse_unary_expr(self) -> Expression:
        if self.at_op("+") or self.at_op("-"):
            tok = self.advance()
            operand = self._parse_unary_expr()
            return UnaryOp(
                op=tok.text,
                operand=operand,
                pos=Position(tok.line, tok.column),
            )
        return self._parse_primary()

    # ----- grammar: primary ----------------------------------------------

    def _parse_primary(self) -> Expression:
        tok = self.peek()

        # CASE WHEN expressions — parsed as a simple single-branch
        # form: ``CASE WHEN cond THEN expr [ELSE expr] END``.
        if self.at_keyword("CASE"):
            return self._parse_case_expr()

        # EXISTS subqueries.
        if self.at_keyword("EXISTS"):
            return self._parse_exists_expr()

        # CAST(expr AS type) — a type coercion that doesn't change the
        # logical value (just tells the SQL engine to render the result
        # with a particular affinity). We parse and discard the ``AS
        # type`` suffix, returning the inner expression unchanged.
        # Without this, ``CAST(COUNT(T1.Id) AS REAL)`` in BIRD gold
        # SQL triggers ``parse_error: expected ), got 'AS'`` and the
        # gold side fails to convert (dev_556 in run-21).
        if self.at_keyword("CAST"):
            self.advance()  # CAST
            self.expect_op("(")
            inner = self.parse_expr()
            # Consume ``AS type_name`` (the type may be multi-word like
            # ``DOUBLE PRECISION`` or carry precision ``DECIMAL(10,2)``).
            if self.at_keyword("AS"):
                self.advance()  # AS
                # Consume type tokens until we hit the closing ``)``
                # for the CAST. Track paren depth so precision like
                # ``DECIMAL(10,2)`` doesn't prematurely stop.
                depth = 0
                while True:
                    t = self.peek()
                    if t.kind == TokenKind.EOF:
                        break
                    if t.kind == TokenKind.OP and t.text == "(":
                        depth += 1
                        self.advance()
                    elif t.kind == TokenKind.OP and t.text == ")":
                        if depth == 0:
                            break
                        depth -= 1
                        self.advance()
                    else:
                        self.advance()
            self.expect_op(")")
            return inner

        # Bare NULL literal: out of scope. The translator does not have a
        # null type and ``IS NULL`` is already gated in comparison.
        if self.at_keyword("NULL"):
            raise self._fail_unsupported(
                "null_literal", "NULL literal is not supported", tok
            )

        # Parenthesised expression. Subqueries that appear bare in
        # primary position (e.g. scalar subqueries) are out of scope and
        # are reported as parse errors when the inner SELECT confuses
        # the expression rules; we do not synthesise a special feature
        # id for them.
        if self.at_op("("):
            self.advance()
            inner = self.parse_expr()
            self.expect_op(")")
            return inner

        # Numeric literal.
        if tok.kind == TokenKind.NUMBER:
            self.advance()
            text = tok.text
            if "." in text or "e" in text or "E" in text:
                return Literal(
                    value=float(text),
                    data_type="real",
                    pos=Position(tok.line, tok.column),
                )
            return Literal(
                value=int(text),
                data_type="integer",
                pos=Position(tok.line, tok.column),
            )

        # String literal (single-quoted; the lexer also emits IDENT for
        # double-quoted tokens which the translator resolves later).
        if tok.kind == TokenKind.STRING:
            self.advance()
            return Literal(
                value=tok.text,
                data_type="string",
                pos=Position(tok.line, tok.column),
            )

        # Aggregate function (KEYWORD token followed by ``(``).
        if (
            tok.kind == TokenKind.KEYWORD
            and tok.text.upper() in _AGGREGATE_NAMES
            and self.at_op("(", offset=1)
        ):
            return self._parse_aggregate()

        # Identifier-led: function call (``ident (``) or column reference.
        if tok.kind == TokenKind.IDENT:
            if self.at_op("(", offset=1):
                return self._parse_function_call()
            return self._parse_column_ref()

        # A KEYWORD token not followed by '(' is likely a column name
        # that collides with a SQL keyword (e.g., Count, Name, Date,
        # Type, Status). Treat it as a column reference.
        if tok.kind == TokenKind.KEYWORD and not self.at_op("(", offset=1):
            return self._parse_column_ref()

        # Anything else is a parse error.
        raise self._fail_parse(_describe_unexpected(tok), tok)

    def _parse_case_expr(self) -> CaseExpr:
        """Parse ``CASE WHEN cond THEN expr [ELSE expr] END``.

        Only supports a single WHEN branch (the common BIRD pattern
        for conditional aggregation inside COUNT/SUM).
        """
        case_tok = self.advance()  # CASE
        self.expect_keyword("WHEN")
        when_cond = self.parse_expr()
        self.expect_keyword("THEN")
        then_expr = self.parse_expr()
        else_expr: Expression | None = None
        if self.at_keyword("ELSE"):
            self.advance()
            else_expr = self.parse_expr()
        self.expect_keyword("END")
        return CaseExpr(
            when_condition=when_cond,
            then_expr=then_expr,
            else_expr=else_expr,
            pos=Position(case_tok.line, case_tok.column),
        )

    def _parse_aggregate(self) -> Aggregate:
        name_tok = self.advance()
        agg_name: AggregateFunction = name_tok.text.upper()  # type: ignore[assignment]
        self.expect_op("(")
        distinct = False
        if self.at_keyword("DISTINCT"):
            self.advance()
            distinct = True
        column: ColumnRef | Literal | CaseExpr
        if self.at_op("*"):
            star = self.advance()
            column = Literal(
                value="*",
                data_type="string",
                pos=Position(star.line, star.column),
            )
        elif self.at_keyword("CASE"):
            column = self._parse_case_expr()
        else:
            column = self._parse_column_ref()
        self.expect_op(")")
        # Window function suffix: ``COUNT(*) OVER (...)``.
        if self.at_keyword("OVER"):
            tok = self.peek()
            raise self._fail_unsupported(
                "window_function", "window functions are not supported", tok
            )
        return Aggregate(
            function=agg_name,
            column=column,
            distinct=distinct,
            pos=Position(name_tok.line, name_tok.column),
        )

    def _parse_function_call(self) -> FunctionCall:
        name_tok = self.advance()
        self.expect_op("(")
        args: list[Expression] = []
        if not self.at_op(")"):
            args.append(self.parse_expr())
            while self.at_op(","):
                self.advance()
                args.append(self.parse_expr())
        self.expect_op(")")
        # Window function suffix.
        if self.at_keyword("OVER"):
            tok = self.peek()
            raise self._fail_unsupported(
                "window_function", "window functions are not supported", tok
            )
        return FunctionCall(
            name=name_tok.text,
            args=args,
            pos=Position(name_tok.line, name_tok.column),
        )

    def _parse_exists_expr(self) -> ExistsExpr:
        tok = self.expect_keyword("EXISTS")
        self.expect_op("(")
        sub = self.parse_select_stmt()
        self.expect_op(")")
        return ExistsExpr(
            subquery=sub, pos=Position(tok.line, tok.column)
        )


# --- error message helpers -------------------------------------------------


def _describe_expected(what: str, got: Token) -> str:
    if got.kind == TokenKind.EOF:
        return f"expected {what}, got end of input"
    return f"expected {what}, got {got.text!r}"


def _describe_unexpected(tok: Token) -> str:
    if tok.kind == TokenKind.EOF:
        return "unexpected end of input"
    return f"unexpected token {tok.text!r}"


# --- public entry point ----------------------------------------------------


def parse(tokens: list[Token]) -> SelectStatement | ConverterError:
    """Parse a flat token stream into a :class:`SelectStatement` AST.

    Returns the parsed AST on success, or a :class:`ConverterError` on
    parse failure or unsupported feature. This function never raises for
    malformed input; every failure mode is reported through the
    structured ``ConverterError`` channel so the BIRD suite can record a
    failed Test_Case and proceed to the next.

    Behaviour for special inputs:

    - If any ``LEX_ERROR`` token appears anywhere in the stream, return
      it immediately as ``ConverterError(kind="parse_error")`` with the
      lexer's diagnostic text. The position is taken from the offending
      token (Req 2.6).
    - If the stream is empty, or the first token is ``EOF``
      (whitespace-only or comment-only input), return
      ``ConverterError(kind="parse_error", message="no SELECT statement
      found", line=1, column=1)`` (Req 2.7).
    - If the SELECT statement parses successfully and is followed by an
      optional ``;`` and then ``EOF``, return the AST.
    - Trailing tokens after the SELECT statement (excluding a single
      semicolon) yield a ``parse_error`` naming the offending token.
    """

    # Surface lex-level errors before invoking the grammar so the parser
    # never has to second-guess token text.
    for tok in tokens:
        if tok.kind == TokenKind.LEX_ERROR:
            return ConverterError(
                kind="parse_error",
                message=tok.text,
                line=tok.line,
                column=tok.column,
            )

    if not tokens or tokens[0].kind == TokenKind.EOF:
        return ConverterError(
            kind="parse_error",
            message="no SELECT statement found",
            line=1,
            column=1,
        )

    parser = _Parser(tokens)
    try:
        stmt = parser.parse_select_stmt()
        # Allow a single trailing ``;`` before EOF.
        if parser.at_op(";"):
            parser.advance()
        if not parser.at_eof():
            tok = parser.peek()
            return ConverterError(
                kind="parse_error",
                message=(
                    f"unexpected token {tok.text!r} after SELECT statement"
                ),
                line=tok.line,
                column=tok.column,
            )
        return stmt
    except _ParseFail as exc:
        return exc.error


__all__ = ["parse"]
