"""Public entry point for the SQL→DRC converter pipeline.

The converter is composed of three stages:

1. :func:`bird_benchmark.sql_to_drc.lexer.tokenize` turns SQL text into a
   flat list of :class:`~bird_benchmark.sql_to_drc.lexer.Token` records
   with 1-indexed source coordinates.
2. :func:`bird_benchmark.sql_to_drc.parser.parse` consumes those tokens
   and emits a :class:`~bird_benchmark.sql_to_drc.ast.SelectStatement`
   AST, gating the input against the Supported_SQL_Subset.
3. :func:`bird_benchmark.sql_to_drc.translator.translate` walks the AST
   and produces a
   :class:`~text_to_sql_planner.types.drc.QueryExpression` (a
   ``DRCExpression`` optionally wrapped in
   :class:`~text_to_sql_planner.types.drc.OrderByExpression` /
   :class:`~text_to_sql_planner.types.drc.LimitExpression`).

Each stage may produce a :class:`~bird_benchmark.types.ConverterError`
instead of advancing — for example a malformed token, an out-of-scope
SQL feature, or an unbound column reference. The public
:func:`convert_sql` entrypoint short-circuits at the first such error
and returns it. It never raises for malformed SQL: the BIRD suite
relies on the structured-error channel so a single Test_Case's bad gold
query does not abort the whole run (Req 2.2, Req 2.3).

The full Supported_SQL_Subset and the list of explicitly out-of-scope
constructs live in the module docstring of
:mod:`bird_benchmark.sql_to_drc.parser` (Req 2.4); this file
deliberately does not duplicate that list so the two cannot drift.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import QueryExpression

from ..types import ConverterError
from .lexer import tokenize
from .parser import parse
from .translator import translate


def convert_sql(sql: str, schema: str) -> QueryExpression | ConverterError:
    """Convert a SQLite SELECT into a DRC query expression.

    Parameters
    ----------
    sql:
        The SQL source text. May be empty, whitespace-only, or
        comment-only — those cases return a structured ``parse_error``
        ConverterError per Req 2.7.
    schema:
        A CREATE-TABLE schema string in the shape
        :class:`bird_benchmark.loader.BirdLoader` produces (one
        ``CREATE TABLE`` block per table, joined by ``;\\n``). The
        translator parses this internally so callers pass a single
        schema string rather than a pre-parsed type map. The translator
        uses the result for SQLite type-affinity decisions (``/``
        integer-vs-real division, double-quoted token resolution). An
        empty string is accepted; the translator falls back to
        ``"String"`` for any unknown column.

    Returns
    -------
    QueryExpression
        A :class:`~text_to_sql_planner.types.drc.DRCExpression`
        optionally wrapped in
        :class:`~text_to_sql_planner.types.drc.OrderByExpression` and/or
        :class:`~text_to_sql_planner.types.drc.LimitExpression` when the
        source had ``ORDER BY`` / ``LIMIT`` clauses (Req 2.2).
    ConverterError
        A structured error describing the first failure encountered:
        ``parse_error`` for malformed SQL or empty/whitespace/comment-only
        input (Req 2.6, Req 2.7), ``unsupported_feature`` for valid SQL
        outside the Supported_SQL_Subset (Req 2.3, Req 13.8), or
        ``unbound_reference`` for a column reference the translator
        cannot resolve in any active scope.

    This function never raises for malformed SQL — every failure mode
    flows through the ``ConverterError`` return value. To honour the
    framework's "never raise" contract (Req 2.3 / Req 3, the suite must
    keep running after one bad gold query), any unexpected exception
    raised inside the lexer, parser, or translator is caught here and
    converted into a ``parse_error`` ConverterError that names the
    underlying exception type and message. Programmer-level errors that
    indicate a bug in the converter itself are preserved verbatim in
    the message so they remain debuggable, but they will not abort a
    benchmark run.
    """

    try:
        tokens = tokenize(sql)
        parsed = parse(tokens)
        if isinstance(parsed, ConverterError):
            return parsed
        return translate(parsed, schema)
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        # The lexer/parser/translator are designed to surface every
        # data-driven failure through the structured ``ConverterError``
        # return path. Anything that escapes as an exception is either
        # a converter bug or an unforeseen edge case in the input; in
        # either case the BIRD suite must keep running, so we convert
        # it into a ``parse_error`` that records the exception type and
        # message at line 1 column 1 (we have no better position
        # information once the structured channel has been bypassed).
        return ConverterError(
            kind="parse_error",
            message=f"converter raised {type(exc).__name__}: {exc}",
            line=1,
            column=1,
        )


__all__ = ["convert_sql"]
