"""Tests for the SQL simplifier rules added in this round.

Two new generally-applicable rules:

1. ``drop_subquery_orderby_without_limit`` — ``ORDER BY`` inside a
   subquery without ``LIMIT`` has no observable effect on the outer
   query (relational SQL is set-based at the boundary). Dropping it
   normalises the AST and unblocks subsequent rewrites that bail when
   they see a non-empty inner ``ORDER BY``.

2. ``count_over_one_row_subquery`` — ``SELECT COUNT(...) FROM (<inner
   that returns exactly one row>) sub`` collapses to ``SELECT 1``
   when the ``COUNT`` argument is statically non-null. Two safe
   shapes: ``COUNT(*)`` and ``COUNT(sub.X)`` where the inner column
   matching ``X`` is itself a ``COUNT(...)`` aggregate (always
   non-null).

Each test phrases its expected behaviour in terms of structural
properties of the simplified AST (rather than exact rendered text)
so cosmetic formatting changes don't break the tests.
"""

from __future__ import annotations

from text_to_sql_planner.sql.simplifier import (
    SqlSelect,
    SqlSubquery,
    SqlTable,
    parse_sql,
    simplify_ast,
    simplify_sql,
)


# ---------------------------------------------------------------------------
# drop_subquery_orderby_without_limit
# ---------------------------------------------------------------------------


def test_drop_orderby_in_from_subquery_without_limit():
    sql = (
        "SELECT s.x FROM ("
        " SELECT x FROM T ORDER BY x"
        ") AS s"
    )
    out = simplify_ast(parse_sql(sql))
    # The simplifier may further inline the passthrough; the
    # invariant we care about is "the resulting AST has no surviving
    # ORDER BY anywhere".
    assert _gather_order_bys(out) == []


def test_keep_orderby_when_limit_present():
    """ORDER BY paired with LIMIT must NOT be dropped — the pair
    selects which rows survive into the outer query.
    """
    sql = (
        "SELECT s.x FROM ("
        " SELECT x FROM T ORDER BY x LIMIT 5"
        ") AS s"
    )
    out = simplify_ast(parse_sql(sql))
    order_bys = _gather_order_bys(out)
    assert "x" in order_bys


def test_keep_top_level_orderby():
    """Outer / top-level ORDER BY is user-visible — never dropped."""
    sql = "SELECT x FROM T ORDER BY x"
    out = simplify_ast(parse_sql(sql))
    assert out.order_by.strip() == "x"


def test_drop_orderby_in_join_subquery_without_limit():
    sql = (
        "SELECT a.x FROM A a "
        "JOIN (SELECT y FROM B ORDER BY y) AS b ON a.x = b.y"
    )
    out = simplify_ast(parse_sql(sql))
    assert _gather_order_bys(out) == []


# ---------------------------------------------------------------------------
# count_over_one_row_subquery
# ---------------------------------------------------------------------------


def test_count_star_over_aggregate_subquery_collapses_to_1():
    """``SELECT COUNT(*) FROM (SELECT COUNT(emp_id) FROM Employees) AS s``
    → ``SELECT 1``. The inner is a scalar-aggregate query (no GROUP
    BY → exactly one row), and ``COUNT(*)`` over a one-row source is 1.
    """
    sql = (
        "SELECT COUNT(*) "
        "FROM (SELECT COUNT(emp_id) FROM Employees) AS s"
    )
    out = simplify_ast(parse_sql(sql))
    assert out.from_source is None
    assert out.columns == ["1"]


def test_count_qualified_over_count_inner_collapses_to_1():
    """The user's exact example from the conversation:
    ``SELECT COUNT(s2.emp_id) FROM (SELECT COUNT(emp_id) FROM ... ) AS s2``.

    The inner column is a ``COUNT(...)`` aggregate (always non-null),
    so the outer ``COUNT(s2.emp_id)`` over the one-row inner reduces
    to 1.

    Note: the user's example referenced ``s2.emp_id`` even though the
    inner column was unnamed. We test the well-formed variant where
    the inner column is exposed as ``emp_id`` via either
    ``COUNT(emp_id) AS emp_id`` (alias) or by relying on the bare
    aggregate exposing a column the outer references.
    """
    sql = (
        "SELECT COUNT(s2.cnt) "
        "FROM (SELECT COUNT(emp_id) AS cnt FROM Employees) AS s2"
    )
    out = simplify_ast(parse_sql(sql))
    assert out.from_source is None
    assert out.columns == ["1"]


def test_count_qualified_over_sum_inner_NOT_collapsed():
    """``SUM`` can be NULL over an empty source, so ``COUNT(s.total)``
    over a one-row source could legitimately be 0. The rule must NOT
    fire — the collapse to 1 would be unsound.
    """
    sql = (
        "SELECT COUNT(s.total) "
        "FROM (SELECT SUM(amount) AS total FROM Sales) AS s"
    )
    out = simplify_ast(parse_sql(sql))
    # The outer FROM still exists (we didn't collapse).
    assert isinstance(out.from_source, SqlSubquery)
    assert "COUNT" in out.columns[0].upper()


def test_count_with_outer_where_NOT_collapsed():
    """An outer WHERE could filter the single inner row out, changing
    the count to 0. The rule must NOT fire.
    """
    sql = (
        "SELECT COUNT(*) "
        "FROM (SELECT COUNT(emp_id) AS n FROM Employees) AS s "
        "WHERE s.n > 100"
    )
    out = simplify_ast(parse_sql(sql))
    assert isinstance(out.from_source, SqlSubquery)
    assert out.where  # WHERE preserved


def test_count_with_outer_groupby_NOT_collapsed():
    sql = (
        "SELECT COUNT(*) "
        "FROM (SELECT COUNT(emp_id) AS n FROM Employees) AS s "
        "GROUP BY s.n"
    )
    out = simplify_ast(parse_sql(sql))
    assert isinstance(out.from_source, SqlSubquery)
    assert out.group_by


def test_count_alias_preserved():
    sql = (
        "SELECT COUNT(*) AS total "
        "FROM (SELECT COUNT(emp_id) FROM Employees) AS s"
    )
    out = simplify_ast(parse_sql(sql))
    assert out.from_source is None
    assert out.columns == ["1 AS total"]


def test_count_over_grouped_inner_NOT_collapsed():
    """Inner with GROUP BY can return any number of rows — not
    one-row guaranteed. The rule must NOT fire.
    """
    sql = (
        "SELECT COUNT(*) "
        "FROM (SELECT dept_id, COUNT(*) FROM Employees GROUP BY dept_id) AS s"
    )
    out = simplify_ast(parse_sql(sql))
    assert isinstance(out.from_source, SqlSubquery)


def test_count_qualified_unrelated_alias_NOT_collapsed():
    """``COUNT(other.x)`` with a qualifier that isn't the subquery's
    alias mustn't trigger the rule (the column isn't even visible
    here). This is an ill-formed query but the simplifier must not
    crash or collapse it.
    """
    sql = (
        "SELECT COUNT(other.x) "
        "FROM (SELECT COUNT(emp_id) AS cnt FROM Employees) AS s"
    )
    out = simplify_ast(parse_sql(sql))
    assert isinstance(out.from_source, SqlSubquery)


# ---------------------------------------------------------------------------
# Interaction tests — both rules cooperating
# ---------------------------------------------------------------------------


def test_orderby_drop_unblocks_count_collapse():
    """An inner ORDER BY would otherwise block the count-collapse rule
    (because we conservatively check ``inner.order_by`` in
    ``_returns_exactly_one_row`` indirectly via the outer's clauses).
    Dropping the unobservable ORDER BY first lets the count-collapse
    rule fire.
    """
    sql = (
        "SELECT COUNT(*) "
        "FROM (SELECT COUNT(emp_id) AS cnt FROM Employees ORDER BY cnt) AS s"
    )
    rendered = simplify_sql(sql)
    assert "ORDER BY" not in rendered
    # Final form should be the constant SELECT 1.
    assert rendered.strip() == "SELECT 1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gather_order_bys(node: SqlSelect) -> list[str]:
    """Walk the SqlSelect tree and collect every non-empty ORDER BY
    string at every depth (FROM source and JOIN sources).

    Used to assert that the simplifier removed every reachable
    ``ORDER BY`` even when the FROM/JOIN subqueries weren't merged
    into the outer.
    """
    out: list[str] = []
    if node.order_by:
        out.append(node.order_by.strip())
    if isinstance(node.from_source, SqlSubquery):
        out.extend(_gather_order_bys(node.from_source.query))
    for j in node.joins:
        if isinstance(j.source, SqlSubquery):
            out.extend(_gather_order_bys(j.source.query))
    return out
