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



# ---------------------------------------------------------------------------
# unwrap_join_subquery_with_where
# ---------------------------------------------------------------------------
#
# ``JOIN (SELECT cols FROM T WHERE p) AS sub ON cond``
# →
# ``JOIN T sub ON cond AND p``
#
# Lifts the inner WHERE up into the join's ON clause so the derived
# subquery disappears. Soundness: ``A JOIN (σ_p T) ON cond ≡ A JOIN T
# ON cond ∧ p`` because both sides produce the multiset
# ``{(a, t) | a ∈ A ∧ t ∈ T ∧ p(t) ∧ cond(a, t)}``.


def test_join_subquery_with_where_unwraps():
    """The user-provided example: a count over Employees joined with a
    WHERE-filtered Locations subquery."""
    sql = (
        "SELECT COUNT(e1.emp_id) "
        "FROM Employees e1 "
        "JOIN ("
        "SELECT l1.location_id FROM Locations l1 "
        "WHERE l1.location_name = 'San Francisco'"
        ") s1 ON e1.location_id = s1.location_id"
    )
    result = simplify_sql(sql)
    # The derived subquery is gone — no ``SELECT`` inside parens.
    assert "(SELECT" not in result.replace("\n", "").replace(" ", "")
    # The inner table is now joined directly.
    assert "JOIN Locations" in result
    # The lifted predicate appears either in WHERE or in the ON clause.
    assert "location_name" in result
    assert "'San Francisco'" in result


def test_join_subquery_with_where_no_outer_where_initially():
    """When the outer query has no WHERE, the lifted predicate should
    still appear (in WHERE or ON)."""
    sql = (
        "SELECT t1.a "
        "FROM T1 t1 "
        "JOIN (SELECT t2.b FROM T2 t2 WHERE t2.b > 5) s ON t1.a = s.b"
    )
    result = simplify_sql(sql)
    ast = parse_sql(result)
    # FROM source is the bare T1 table.
    assert isinstance(ast.from_source, SqlTable)
    assert ast.from_source.name == "T1"
    # One JOIN remains, against the bare T2 table (alias ``s``).
    assert len(ast.joins) == 1
    assert isinstance(ast.joins[0].source, SqlTable)
    assert ast.joins[0].source.name == "T2"
    # Predicate was lifted somewhere.
    full = result.lower()
    assert "b > 5" in full


def test_join_subquery_with_where_combines_with_outer_where():
    """An existing outer WHERE conjoins with the lifted predicate
    (after the lift-singletable rule moves it from ON to WHERE)."""
    sql = (
        "SELECT t1.a "
        "FROM T1 t1 "
        "JOIN (SELECT t2.b FROM T2 t2 WHERE t2.b > 5) s ON t1.a = s.b "
        "WHERE t1.a < 100"
    )
    result = simplify_sql(sql)
    full = result.lower()
    # Both predicates present.
    assert "b > 5" in full
    assert "a < 100" in full


def test_join_subquery_no_where_uses_passthrough_path():
    """Without an inner WHERE, the older passthrough rule fires —
    the new with-where rule must NOT double-apply."""
    sql = (
        "SELECT t1.a "
        "FROM T1 t1 "
        "JOIN (SELECT t2.b FROM T2 t2) s ON t1.a = s.b"
    )
    result = simplify_sql(sql)
    ast = parse_sql(result)
    # Single join against a bare table.
    assert isinstance(ast.joins[0].source, SqlTable)
    assert ast.joins[0].source.name == "T2"


def test_join_subquery_with_groupby_not_unwrapped():
    """Inner GROUP BY blocks the lift: aggregating before the join
    changes which rows are present, can't be commuted with the outer
    join."""
    sql = (
        "SELECT t1.a "
        "FROM T1 t1 "
        "JOIN ("
        "SELECT t2.b, COUNT(*) FROM T2 t2 WHERE t2.b > 5 GROUP BY t2.b"
        ") s ON t1.a = s.b"
    )
    result = simplify_sql(sql)
    ast = parse_sql(result)
    # Subquery preserved.
    assert isinstance(ast.joins[0].source, SqlSubquery)


def test_join_subquery_with_aggregate_not_unwrapped():
    """Inner aggregate without GROUP BY (scalar aggregate) returns one
    row — completely different shape from the bare table. Don't lift."""
    sql = (
        "SELECT t1.a "
        "FROM T1 t1 "
        "JOIN (SELECT COUNT(*) AS c FROM T2 t2 WHERE t2.b > 5) s ON t1.a = s.c"
    )
    result = simplify_sql(sql)
    ast = parse_sql(result)
    assert isinstance(ast.joins[0].source, SqlSubquery)


# ---------------------------------------------------------------------------
# lift_singletable_join_predicates
# ---------------------------------------------------------------------------
#
# ``JOIN T t ON p1 AND p2`` where ``p2`` references only one alias
# →
# ``JOIN T t ON p1 WHERE p2``
#
# Soundness for INNER JOIN: predicate placement is interchangeable;
# the multiset of joined rows is identical. NOT applied to OUTER JOINs
# because ``ON`` filters before NULL-extension while ``WHERE`` filters
# after.


def test_lift_filter_from_inner_join_on_to_where():
    """Single-table predicate in ON moves to WHERE."""
    sql = (
        "SELECT e1.emp_id "
        "FROM Employees e1 "
        "JOIN Locations l1 ON e1.location_id = l1.location_id "
        "AND l1.location_name = 'San Francisco'"
    )
    result = simplify_sql(sql)
    ast = parse_sql(result)
    # The join's ON clause kept only the multi-table predicate.
    assert "location_name" not in ast.joins[0].on_condition.lower()
    # The single-table predicate moved to WHERE.
    assert "location_name" in ast.where.lower()


def test_lift_does_not_touch_multitable_predicates():
    """A predicate referencing both sides of the join stays in ON."""
    sql = (
        "SELECT e1.emp_id "
        "FROM Employees e1 "
        "JOIN Compensation c1 ON e1.emp_id = c1.emp_id"
    )
    result = simplify_sql(sql)
    ast = parse_sql(result)
    # The join predicate stayed in ON; WHERE is empty.
    assert "e1.emp_id" in ast.joins[0].on_condition
    assert "c1.emp_id" in ast.joins[0].on_condition
    assert not ast.where.strip()


def test_lift_handles_quoted_string_with_dot():
    """A predicate involving a string literal containing a dot must NOT
    confuse the alias-detector. Email-like literals are the common case."""
    sql = (
        "SELECT e1.emp_id "
        "FROM Employees e1 "
        "JOIN Locations l1 ON e1.location_id = l1.location_id "
        "AND l1.contact = 'a.b@x.com'"
    )
    result = simplify_sql(sql)
    ast = parse_sql(result)
    # The single-table predicate moved to WHERE despite the dotted
    # literal.
    assert "contact" in ast.where.lower()
    assert "'a.b@x.com'" in ast.where
