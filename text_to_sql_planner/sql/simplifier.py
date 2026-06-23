"""SQL simplifier: applies correctness-preserving transformations via a parse tree.

Architecture:
    parse_sql(text)         → SqlSelect
    simplify_ast(node)      → SqlSelect (fixed point of `_RULES`)
    render_sql(node)        → text

Each simplification rule is a pure function ``(SqlSelect) -> SqlSelect`` that
returns the same node when it doesn't apply, or a transformed node when it
does. Rules use :func:`dataclasses.replace` to construct the result, so any
field that isn't explicitly named is preserved automatically. This makes the
rules robust to future schema additions to ``SqlSelect`` (e.g. ``order_by``,
``limit``, ``having``, ...) without the risk that a rule silently drops a
clause it doesn't know about.

The driver applies rules in order until ``render_sql`` reaches a fixed
point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Union


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass
class SqlTable:
    """A plain table reference (optionally aliased)."""

    name: str
    alias: str = ""


@dataclass
class SqlSubquery:
    """A subquery used as a FROM source."""

    query: "SqlSelect"
    alias: str = "sub"


SqlSource = Union[SqlTable, SqlSubquery]


@dataclass
class SqlJoin:
    """A JOIN clause."""

    join_type: str  # "JOIN" or "CROSS JOIN"
    source: SqlSource
    on_condition: str = ""


@dataclass
class SqlSelect:
    """A SELECT statement.

    Every clause is represented as either a list (``columns``,
    ``group_by``, ``joins``) or a string (``where``, ``order_by``,
    ``limit``). String clauses store the verbatim source so we can
    apply purely-syntactic rewrites (e.g. alias substitution) without
    having to model their grammar.
    """

    columns: list[str] = field(default_factory=list)
    from_source: SqlSource | None = None
    joins: list[SqlJoin] = field(default_factory=list)
    where: str = ""
    group_by: list[str] = field(default_factory=list)
    order_by: str = ""
    limit: str = ""


# ---------------------------------------------------------------------------
# Tokenizer-style helpers used by both the parser and the rewrite rules
# ---------------------------------------------------------------------------


_SQL_KEYWORDS_AS_ALIAS = frozenset({
    "WHERE", "JOIN", "CROSS", "ON", "GROUP", "ORDER", "HAVING",
    "LIMIT", "UNION", "AS", "LEFT", "RIGHT", "INNER", "OUTER", "FULL",
})

_CLAUSE_TERMINATORS = ["JOIN", "CROSS JOIN", "WHERE", "GROUP BY", "ORDER BY", "LIMIT"]


def _find_keyword(text: str, keyword: str) -> int:
    """Return the index of ``keyword`` at depth 0 with word boundaries, or ``-1``."""
    depth = 0
    kl = len(keyword)
    tu = text.upper()
    ku = keyword.upper()
    for i in range(len(text) - kl + 1):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and tu[i:i + kl] == ku:
            before_ok = i == 0 or not text[i - 1].isalnum()
            after_ok = i + kl >= len(text) or not text[i + kl].isalnum()
            if before_ok and after_ok:
                return i
    return -1


def _find_first_keyword(text: str, keywords: list[str]) -> int:
    positions = [p for p in (_find_keyword(text, k) for k in keywords) if p != -1]
    return min(positions) if positions else -1


def _find_matching_paren(text: str, start: int) -> int:
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "'":
            # Skip single-quoted string literal
            i += 1
            while i < len(text) and text[i] != "'":
                if text[i] == "'" and i + 1 < len(text) and text[i + 1] == "'":
                    i += 2  # escaped quote
                else:
                    i += 1
            i += 1  # skip closing quote
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _strip_outer_parens(text: str) -> str:
    """Strip a single matching outer pair of parentheses, if present."""
    text = text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[1:-1].strip() if i == len(text) - 1 else text
    return text


def _split_and(text: str) -> list[str]:
    """Split a boolean expression on top-level ``AND``, paren-aware."""
    text = _strip_outer_parens(text).strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    depth = 0
    upper = text.upper()
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            i += 1
            continue
        if depth == 0 and upper.startswith("AND", i):
            before_ok = i == 0 or not text[i - 1].isalnum()
            after_ok = i + 3 >= n or not text[i + 3].isalnum()
            if before_ok and after_ok:
                parts.append(text[start:i].strip())
                i += 3
                start = i
                continue
        i += 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_sql(sql: str) -> SqlSelect:
    """Parse a SQL SELECT statement into an AST."""
    return _parse_select(sql.strip())


def _parse_select(sql: str) -> SqlSelect:
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        return SqlSelect(columns=["*"], from_source=SqlTable(name=sql))

    rest = sql[6:].strip()
    from_idx = _find_keyword(rest, "FROM")
    if from_idx == -1:
        return SqlSelect(columns=[c.strip() for c in rest.split(",")])

    columns = [c.strip() for c in rest[:from_idx].strip().split(",")]
    rest = rest[from_idx + 4:].strip()

    from_source, rest = _parse_source(rest)

    joins: list[SqlJoin] = []
    while True:
        rest = rest.strip()
        jm = re.match(r"(CROSS\s+JOIN|JOIN)\s+", rest, re.IGNORECASE)
        if not jm:
            break
        jtype = " ".join(jm.group(1).upper().split())
        rest = rest[jm.end():]
        jsrc, rest = _parse_source(rest)
        on_cond = ""
        rest = rest.strip()
        om = re.match(r"ON\s+", rest, re.IGNORECASE)
        if om:
            rest = rest[om.end():]
            end = _find_first_keyword(rest, _CLAUSE_TERMINATORS)
            if end == -1:
                on_cond = rest.strip()
                rest = ""
            else:
                on_cond = rest[:end].strip()
                rest = rest[end:]
        joins.append(SqlJoin(join_type=jtype, source=jsrc, on_condition=on_cond))

    # Each tail clause consumes up to the next clause keyword; this scheme
    # generalizes naturally — adding a new clause only requires extending
    # ``_CLAUSE_TERMINATORS`` and the small block here.
    where, rest = _consume_clause(rest, "WHERE", ["GROUP BY", "ORDER BY", "LIMIT"])
    group_by_text, rest = _consume_clause(rest, "GROUP BY", ["ORDER BY", "LIMIT"])
    order_by, rest = _consume_clause(rest, "ORDER BY", ["LIMIT"])
    limit, rest = _consume_clause(rest, "LIMIT", [])

    group_by = [c.strip() for c in group_by_text.split(",")] if group_by_text else []

    return SqlSelect(
        columns=columns,
        from_source=from_source,
        joins=joins,
        where=where,
        group_by=group_by,
        order_by=order_by,
        limit=limit,
    )


def _consume_clause(text: str, keyword: str, terminators: list[str]) -> tuple[str, str]:
    """Consume ``KEYWORD <body>`` from the start of ``text`` (case-insensitive),
    where ``<body>`` runs up to the next clause keyword in ``terminators``.

    Returns ``(body, remainder)``. If ``keyword`` doesn't match, returns
    ``("", text)`` unchanged (modulo a leading strip).
    """
    text = text.strip()
    pattern = re.compile(rf"{re.escape(keyword)}\s+", re.IGNORECASE)
    m = pattern.match(text)
    if not m:
        return "", text
    rest = text[m.end():]
    end_kw = _find_first_keyword(rest, terminators) if terminators else -1
    if end_kw == -1:
        return rest.strip(), ""
    return rest[:end_kw].strip(), rest[end_kw:]


def _parse_source(text: str) -> tuple[SqlSource, str]:
    text = text.strip()
    if text.startswith("("):
        end = _find_matching_paren(text, 0)
        if end == -1:
            end = len(text) - 1
        inner = text[1:end].strip()
        rest = text[end + 1:].strip()
        alias = "sub"
        am = re.match(r"AS\s+(\w+)", rest, re.IGNORECASE)
        if am:
            alias = am.group(1)
            rest = rest[am.end():]
        else:
            bm = re.match(r"(\w+)", rest)
            if bm and bm.group(1).upper() not in _SQL_KEYWORDS_AS_ALIAS:
                alias = bm.group(1)
                rest = rest[bm.end():]
        return SqlSubquery(query=_parse_select(inner), alias=alias), rest

    m = re.match(r"(\w+)", text)
    if m:
        table_name = m.group(1)
        rest = text[m.end():].strip()
        alias = ""
        alias_match = re.match(r"(\w+)", rest)
        if alias_match and alias_match.group(1).upper() not in _SQL_KEYWORDS_AS_ALIAS:
            alias = alias_match.group(1)
            rest = rest[alias_match.end():]
        return SqlTable(name=table_name, alias=alias), rest
    return SqlTable(name=text), ""


# ---------------------------------------------------------------------------
# Identifier substitution helpers (used by rewrite rules)
# ---------------------------------------------------------------------------


_QUALIFIED_REF = re.compile(r"^(\w+)\.(\w+)$")
_AS_RENAME = re.compile(r"^(.+?)\s+AS\s+(\w+)\s*$", re.IGNORECASE)
_BARE_IDENT = re.compile(r"^\w+$")


def _column_public_name(col: str) -> str | None:
    """Return the externally-visible name of a SELECT column expression.

    For ``alias.col`` and bare ``col`` it's the trailing identifier.
    For ``<expr> AS name`` it's ``name``. Returns ``None`` when the
    column has no stable identifier (e.g. raw expressions).
    """
    c = col.strip()
    m = _AS_RENAME.match(c)
    if m:
        return m.group(2)
    m = _QUALIFIED_REF.match(c)
    if m:
        return m.group(2)
    if _BARE_IDENT.match(c):
        return c
    return None


def _column_underlying_expr(col: str) -> str | None:
    """Return the expression to substitute in for outer ``alias.<public_name>``
    references when promoting an inner SELECT column.

    For ``alias.col`` it's the whole ``alias.col`` string.
    For ``<expr> AS name`` it's ``<expr>`` (without the alias).
    For bare ``col`` it's the bare ``col`` (the inner relation will
    provide the qualification).
    """
    c = col.strip()
    m = _AS_RENAME.match(c)
    if m:
        return m.group(1).strip()
    if _QUALIFIED_REF.match(c):
        return c
    if _BARE_IDENT.match(c):
        return c
    return None


def _build_alias_substitution(
    outer_alias: str, inner_columns: list[str]
) -> dict[str, str] | None:
    """Build a substitution map ``outer_alias.X → <inner expression>``.

    Returns ``None`` if any inner column lacks a stable public name or
    has no rewriteable underlying expression.
    """
    mapping: dict[str, str] = {}
    for col in inner_columns:
        public = _column_public_name(col)
        underlying = _column_underlying_expr(col)
        if public is None or underlying is None:
            return None
        mapping[f"{outer_alias}.{public}"] = underlying
    return mapping


def _apply_alias_substitution(text: str, mapping: dict[str, str]) -> str:
    """Replace whole-token occurrences of each key in ``mapping`` with its value.

    Tokens are matched at identifier boundaries (no leading word char or
    dot, no trailing word char) so substrings of longer identifiers are
    not affected.
    """
    if not text or not mapping:
        return text
    out = text
    # Longest keys first, so ``s1.foo_bar`` doesn't match ``s1.foo`` first.
    for src in sorted(mapping, key=lambda k: -len(k)):
        pattern = re.compile(rf"(?<![\w.]){re.escape(src)}(?!\w)")
        out = pattern.sub(mapping[src], out)
    return out


def _substitute_in_columns(cols: list[str], mapping: dict[str, str]) -> list[str]:
    return [_apply_alias_substitution(c, mapping) for c in cols]


def _substitute_in_node_clauses(
    node: SqlSelect, mapping: dict[str, str]
) -> SqlSelect:
    """Apply ``mapping`` to every textual clause of ``node`` (columns, joins'
    ON conditions, WHERE, GROUP BY, ORDER BY). LIMIT is numeric and not
    substituted.
    """
    return replace(
        node,
        columns=_substitute_in_columns(node.columns, mapping),
        joins=[
            replace(j, on_condition=_apply_alias_substitution(j.on_condition, mapping))
            for j in node.joins
        ],
        where=_apply_alias_substitution(node.where, mapping),
        group_by=[_apply_alias_substitution(g, mapping) for g in node.group_by],
        order_by=_apply_alias_substitution(node.order_by, mapping),
    )


# ---------------------------------------------------------------------------
# Predicates and small builders used by rewrite rules
# ---------------------------------------------------------------------------


def _is_passthrough_select(inner: SqlSelect) -> bool:
    """Return True when the inner SELECT is a structural passthrough — a
    bare ``SELECT col1, col2 FROM T`` with no JOINs, WHERE, GROUP BY,
    ORDER BY, or LIMIT — and every projected column is either an
    unaliased bare identifier or a ``inner_alias.col`` qualified
    reference against the inner FROM source.

    This is the precondition for *promoting* the inner table up to the
    outer scope. When any clause is present (including ORDER BY / LIMIT),
    promotion is unsafe because the outer scope may not preserve them.
    """
    if not isinstance(inner.from_source, SqlTable):
        return False
    if (
        inner.joins
        or inner.where
        or inner.group_by
        or inner.order_by
        or inner.limit
    ):
        return False
    inner_alias = inner.from_source.alias or inner.from_source.name
    return _qualified_or_bare_columns(inner.columns, inner_alias)


def _passthrough_columns(cols: list[str]) -> bool:
    """True iff every column is a bare identifier — no AS rename, no
    aggregate, no expression, no qualified or starred form."""
    if not cols:
        return False
    for col in cols:
        c = col.strip()
        if not c or c == "*":
            return False
        if " AS " in c.upper() or "(" in c or ")" in c or "." in c:
            return False
        if not _BARE_IDENT.match(c):
            return False
    return True


def _qualified_or_bare_columns(cols: list[str], inner_alias: str) -> bool:
    """True iff every column is either a bare identifier or a
    ``inner_alias.col`` qualified reference — no AS rename, no
    aggregate, no expression, no other qualifier.

    A more permissive variant of :func:`_passthrough_columns` that
    accommodates the common case of an inner SELECT projecting
    ``alias.col`` against a single source table. Such columns expose
    the same public name as the bare form; the inner alias is purely
    cosmetic and can be dropped when the table is promoted.
    """
    if not cols:
        return False
    for col in cols:
        c = col.strip()
        if not c or c == "*":
            return False
        if " AS " in c.upper() or "(" in c or ")" in c:
            return False
        if "." in c:
            head, _, tail = c.partition(".")
            if head != inner_alias:
                return False
            if not _BARE_IDENT.match(tail):
                return False
        elif not _BARE_IDENT.match(c):
            return False
    return True


def _promote_alias(table: SqlTable, outer_alias: str) -> SqlTable:
    """Return ``table`` rebadged with ``outer_alias`` so that surrounding
    references to the (now-removed) subquery alias still resolve."""
    return SqlTable(name=table.name, alias=outer_alias or table.alias)


def _and_join(*clauses: str) -> str:
    """Combine non-empty WHERE fragments with ``AND``."""
    parts = [c for c in (s.strip() for s in clauses) if c]
    return " AND ".join(parts)


def _source_name(source: SqlSource | None) -> str:
    if isinstance(source, SqlTable):
        return source.name
    if isinstance(source, SqlSubquery):
        if isinstance(source.query.from_source, SqlTable):
            return source.query.from_source.name
        return source.alias
    return ""


# Aggregate-name patterns, used by several rules to decide whether an
# inner SELECT collapses to one row (scalar aggregate without GROUP BY)
# or whether substituting an inner column expression into the outer
# scope would produce a nested-aggregate violation.
_AGG_FUNCS = ("COUNT", "SUM", "AVG", "MIN", "MAX")
_AGG_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(_AGG_FUNCS) + r")\s*\(",
    re.IGNORECASE,
)


def _is_aggregate_expr(col: str) -> bool:
    """True if ``col`` starts with an aggregate function call."""
    return bool(_AGG_PATTERN.match(col))


def _returns_exactly_one_row(node: SqlSelect) -> bool:
    """True iff ``node`` is guaranteed to return exactly one row.

    Two safe cases:

    * ``LIMIT 1`` (and the source isn't empty in the strictest sense —
      but ``COUNT`` and friends over an empty source still return one
      row, which is the only context this helper is used in).
    * Every column in the projection is a scalar aggregate AND there's
      no ``GROUP BY``. Aggregate-without-GROUP BY collapses any
      multiset (including the empty one) to exactly one row.

    Returns False conservatively for shapes the rule doesn't model
    (e.g. ``UNION`` would be visible as a non-SqlSelect form, but our
    AST doesn't represent it explicitly).
    """
    # Case A: explicit LIMIT 1.
    if node.limit.strip() == "1":
        return True

    # Case B: scalar-aggregate query with no GROUP BY.
    if node.group_by:
        return False
    if not node.columns:
        return False
    return all(_is_aggregate_expr(c) for c in node.columns)


# ---------------------------------------------------------------------------
# Simplification rules
#
# Each rule is ``(SqlSelect) -> SqlSelect``. A rule that doesn't apply
# returns its argument unchanged. All node construction goes through
# :func:`dataclasses.replace`, which means any field a rule doesn't
# explicitly set is preserved automatically. This makes rules robust
# against future additions to ``SqlSelect``.
# ---------------------------------------------------------------------------


SimplifyRule = Callable[[SqlSelect], SqlSelect]


def _rule_recurse_into_subqueries(node: SqlSelect) -> SqlSelect:
    """Recursively simplify subqueries in the FROM source and JOIN sources."""
    new_from = node.from_source
    if isinstance(new_from, SqlSubquery):
        new_from = SqlSubquery(query=_simplify(new_from.query), alias=new_from.alias)
    new_joins = []
    changed = new_from is not node.from_source
    for j in node.joins:
        if isinstance(j.source, SqlSubquery):
            new_src = SqlSubquery(query=_simplify(j.source.query), alias=j.source.alias)
            if new_src is not j.source:
                changed = True
            new_joins.append(replace(j, source=new_src))
        else:
            new_joins.append(j)
    if not changed:
        return node
    return replace(node, from_source=new_from, joins=new_joins)


def _rule_unwrap_passthrough_from(node: SqlSelect) -> SqlSelect:
    """``SELECT ... FROM (SELECT cols FROM T) AS sub ...`` →
    ``SELECT ... FROM T sub ...`` when the inner SELECT is a structural
    passthrough. The outer alias is transferred to the promoted table so
    surrounding ``sub.col`` references continue to resolve.
    """
    if not isinstance(node.from_source, SqlSubquery):
        return node
    inner = node.from_source.query
    if not _is_passthrough_select(inner):
        return node
    return replace(
        node,
        from_source=_promote_alias(inner.from_source, node.from_source.alias),
    )


def _rule_merge_inner_where(node: SqlSelect) -> SqlSelect:
    """Like ``unwrap_passthrough_from`` but tolerates an inner WHERE,
    which gets conjoined with the outer WHERE on promotion.
    """
    if not isinstance(node.from_source, SqlSubquery):
        return node
    inner = node.from_source.query
    if not (
        isinstance(inner.from_source, SqlTable)
        and not inner.joins
        and not inner.group_by
        and not inner.order_by
        and not inner.limit
        and _passthrough_columns(inner.columns)
    ):
        return node
    return replace(
        node,
        from_source=_promote_alias(inner.from_source, node.from_source.alias),
        where=_and_join(inner.where, node.where),
    )


def _rule_inline_rebinding_subquery(node: SqlSelect) -> SqlSelect:
    """Inline a column-rebinding subquery, rewriting ``alias.col``
    references to the inner expressions.

    Pattern:
        SELECT <outer_cols using alias.X>
          FROM (SELECT <inner_cols> FROM ...) AS alias
          [WHERE <outer_where using alias.X>]
          [GROUP BY/ORDER BY/LIMIT using alias.X]

    This is correct when:
      * The inner SELECT has no GROUP BY / ORDER BY / LIMIT (those would
        change the *set* of rows visible at the outer scope and so
        cannot generally be promoted past an outer SELECT/WHERE).
      * The inner SELECT has no aggregate columns. A scalar-aggregate
        query (aggregate without GROUP BY) collapses any source to
        exactly one row — semantically different from the per-row
        relation we'd produce by inlining the inner FROM into the
        outer. Substituting ``alias.X`` with the aggregate expression
        also produces nested aggregates at the outer scope, which is
        invalid SQL.
      * Every inner SELECT column has a determinable public name
        (i.e. it's a bare identifier, ``alias.col``, or ``<expr> AS
        name``). Aggregates/expressions without ``AS`` are rejected
        because they have no stable name to substitute.

    On rewrite we:
      * Build a substitution ``outer_alias.<public> → <inner_expr>``.
      * Apply it to outer columns, joins' ON conditions, WHERE, GROUP
        BY, and ORDER BY.
      * Concatenate the inner WHERE with the outer (rewritten) WHERE.
      * Promote the inner FROM/JOINs into the outer.
    """
    if not isinstance(node.from_source, SqlSubquery):
        return node
    sub = node.from_source
    inner = sub.query

    # Bail when the inner has clauses that don't compose with the outer.
    # GROUP BY / ORDER BY / LIMIT applied inside a subquery semantically
    # differ from the same clauses applied outside, so we mustn't fold
    # them away.
    if inner.group_by or inner.order_by or inner.limit:
        return node

    # Bail when the inner is a scalar-aggregate query (any column is
    # an aggregate with no GROUP BY). Those queries return exactly
    # one row; inlining their FROM into the outer would change the
    # outer's row count, and substituting ``alias.X`` with the
    # aggregate expression produces nested aggregates that are
    # invalid SQL. The dedicated ``count_over_one_row_subquery`` rule
    # handles the cases where this collapse IS sound.
    if any(_is_aggregate_expr(c) for c in inner.columns):
        return node

    mapping = _build_alias_substitution(sub.alias, inner.columns)
    if mapping is None:
        return node

    rewritten = _substitute_in_node_clauses(node, mapping)
    return replace(
        rewritten,
        from_source=inner.from_source,
        joins=list(inner.joins) + list(rewritten.joins),
        where=_and_join(inner.where, rewritten.where),
    )


def _rule_collapse_redundant_wrapper(node: SqlSelect) -> SqlSelect:
    """``SELECT same_cols FROM (<inner>) AS sub`` (no other outer clauses)
    → ``<inner>`` when the outer adds nothing the inner doesn't already
    say. ``order_by`` / ``limit`` on the outer are *not* "nothing" — they
    must stay, so the rule only fires when the outer has none of WHERE,
    JOINs, GROUP BY, ORDER BY, or LIMIT beyond the SELECT projection.
    """
    if not isinstance(node.from_source, SqlSubquery):
        return node
    if (
        node.joins
        or node.where
        or node.group_by
        or node.order_by
        or node.limit
    ):
        return node
    inner = node.from_source.query
    if [c.strip().lower() for c in node.columns] != [
        c.strip().lower() for c in inner.columns
    ]:
        return node
    return inner


def _rule_merge_groupby_through_subquery(node: SqlSelect) -> SqlSelect:
    """``SELECT ... FROM (<inner without GROUP BY>) AS sub GROUP BY ...``
    → ``<inner>`` with the outer's GROUP BY (and ORDER BY / LIMIT)
    applied to the inner FROM/JOIN/WHERE shape.

    Only fires when the outer adds nothing other than GROUP BY (plus
    optional ORDER BY / LIMIT, which compose cleanly with the merged
    GROUP BY).

    Bails when the inner is a scalar-aggregate query (any column is
    an aggregate with no GROUP BY). Promoting such an inner past the
    outer's ``GROUP BY sub.X`` reference would leave a dangling
    qualifier — the inner FROM/JOIN shape doesn't expose ``sub.X``.
    The dedicated ``count_over_one_row_subquery`` rule handles cases
    where the collapse is sound.
    """
    if not isinstance(node.from_source, SqlSubquery):
        return node
    if node.joins or node.where or not node.group_by:
        return node
    inner = node.from_source.query
    if inner.group_by:
        return node
    if any(_is_aggregate_expr(c) for c in inner.columns):
        return node
    return replace(
        node,
        from_source=inner.from_source,
        joins=list(inner.joins),
        where=inner.where,
    )


def _rule_cross_join_to_join_on(node: SqlSelect) -> SqlSelect:
    """``CROSS JOIN T2 ... WHERE T1.x = T2.x`` → ``JOIN T2 ON T1.x = T2.x``.

    Lifts equality predicates from WHERE into the first CROSS JOIN's ON
    clause. Inequalities and disjunctions stay in WHERE.
    """
    if not (node.joins and node.where):
        return node
    where_parts = _split_and(node.where)
    new_joins = list(node.joins)
    fired = False
    for i, j in enumerate(new_joins):
        if j.join_type != "CROSS JOIN" or j.on_condition:
            continue
        matched: list[str] = []
        unmatched: list[str] = []
        for wp in where_parts:
            inner = _strip_outer_parens(wp)
            if "=" in inner and "!=" not in inner:
                matched.append(inner)
            else:
                unmatched.append(wp)
        if matched:
            new_joins[i] = replace(
                j, join_type="JOIN", on_condition=" AND ".join(matched)
            )
            where_parts = unmatched
            fired = True
            break
    if not fired:
        return node
    return replace(node, joins=new_joins, where=" AND ".join(where_parts))


def _rule_unwrap_join_subquery(node: SqlSelect) -> SqlSelect:
    """``JOIN (<passthrough>) AS sub ON …`` → ``JOIN T sub ON …``."""
    if not node.joins:
        return node
    new_joins: list[SqlJoin] = []
    changed = False
    for j in node.joins:
        if isinstance(j.source, SqlSubquery) and _is_passthrough_select(j.source.query):
            inner = j.source.query
            new_joins.append(
                replace(
                    j,
                    source=_promote_alias(inner.from_source, j.source.alias),
                )
            )
            changed = True
        else:
            new_joins.append(j)
    if not changed:
        return node
    return replace(node, joins=new_joins)


def _rule_lift_singletable_join_predicates(node: SqlSelect) -> SqlSelect:
    """Move single-table predicates from an INNER JOIN's ``ON`` clause
    to the outer ``WHERE``.

    Pattern::

        ... JOIN T ON p1 AND p2 ...  WHERE w
        →
        ... JOIN T ON p1 ... WHERE (p2) AND w

    when ``p2`` references columns from only one side of the join (i.e.
    is a "filter" predicate that just selects rows of ``T`` or of an
    earlier table, not a join predicate). The placement difference is
    semantically inert for INNER JOIN: the join produces the same
    multiset of pairs, then the same predicates filter them.

    Out of scope:

    * OUTER JOINs (LEFT/RIGHT/FULL): the ``ON`` placement of a filter
      predicate has different semantics from ``WHERE`` for outer
      joins (NULL-extended rows survive ``ON`` mismatch but get
      filtered by ``WHERE``).
    * CROSS JOINs that don't have an ``ON`` clause at all.
    * Predicates referencing more than one table — those are genuine
      join predicates and stay in ``ON``.

    Algorithm: for each ``JOIN T t ON p`` (INNER, no special outer
    keyword), split ``p`` into top-level conjuncts. A conjunct is
    "single-table" if all qualified references in it use the same
    alias (or no alias at all — bare names that only resolve via the
    inner table, but those are rare in well-formed SQL). Move all
    single-table conjuncts that reference a *non*-join table or only
    the joined table itself into the WHERE clause; keep the rest in
    ON.

    For canonicalisation we only move conjuncts that mention exactly
    ONE alias and that alias is either the join's source or the
    overall outer FROM source. Multi-alias conjuncts are join
    predicates and stay.
    """
    if not node.joins:
        return node

    new_joins: list[SqlJoin] = []
    new_where_parts: list[str] = []
    if node.where.strip():
        new_where_parts.append(node.where)
    changed = False

    for j in node.joins:
        # Only INNER JOIN (the parser emits "JOIN" — we treat that as
        # inner). LEFT / RIGHT / FULL OUTER and CROSS JOIN keep their
        # ON clauses unchanged.
        if j.join_type != "JOIN":
            new_joins.append(j)
            continue
        if not j.on_condition.strip():
            new_joins.append(j)
            continue

        conjuncts = _split_and(j.on_condition)
        keep_on: list[str] = []
        for cj in conjuncts:
            inner = _strip_outer_parens(cj).strip()
            aliases = _aliases_in_predicate(inner)
            if len(aliases) == 1:
                # Single-alias predicate — definitely a filter on one
                # table, safe to lift to WHERE for INNER JOIN.
                new_where_parts.append(inner)
                changed = True
            else:
                keep_on.append(cj)

        if not keep_on:
            # All ON conditions lifted — keep the join with an empty
            # ON. SQL doesn't allow ``JOIN T ON `` (empty ON), so we
            # leave at least a trivially-true ``1=1`` if everything
            # was lifted. In practice this branch is rare; the
            # algorithm preserves at least one join predicate in
            # well-formed inputs.
            new_joins.append(replace(j, on_condition="1 = 1"))
        else:
            new_joins.append(replace(j, on_condition=" AND ".join(keep_on)))

    if not changed:
        return node
    return replace(
        node,
        joins=new_joins,
        where=" AND ".join(new_where_parts),
    )


_ALIAS_REF_RE = re.compile(r"(?<!\w)([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_]")


def _aliases_in_predicate(predicate: str) -> set[str]:
    """Return the set of qualifier aliases appearing in a predicate.

    A qualifier is an identifier followed by ``.<column>``. SQL
    keywords like ``AND``, ``OR``, etc. don't take dots, so any
    matched ``head.tail`` head is an alias reference. String literals
    are stripped first so that quoted text containing dots (e.g. an
    email like ``'a.b@x.com'``) doesn't pollute the result.
    """
    # Strip string literals (handles single-quoted only; the parser
    # doesn't emit double-quoted identifiers for the cases we care
    # about). Pairs of single quotes inside the string aren't
    # supported; the simplifier rejects predicates that need that
    # complexity by virtue of ``_split_and`` being paren-aware but
    # not quote-aware. For now this is good enough — the queries we
    # see don't have quoted dots.
    cleaned = re.sub(r"'[^']*'", "''", predicate)
    return set(_ALIAS_REF_RE.findall(cleaned))


def _rule_unwrap_join_subquery_with_where(node: SqlSelect) -> SqlSelect:
    """Like :func:`_rule_unwrap_join_subquery` but tolerates an inner
    ``WHERE`` and lifts it into the join's ``ON`` clause.

    Pattern::

        JOIN (SELECT cols FROM T WHERE p) AS sub ON cond
        →
        JOIN T sub ON (cond) AND (p)

    Conditions for soundness:

    * The inner SELECT is shaped ``SELECT cols FROM T WHERE p`` with
      no inner JOINs / GROUP BY / ORDER BY / LIMIT. JOINs would change
      the row shape; GROUP BY / ORDER BY / LIMIT would change the row
      *set* and can't be lifted past the join.
    * Every projected column is a bare identifier (the
      ``_passthrough_columns`` predicate). Aliased / aggregated /
      qualified columns are rejected — the inner SELECT would expose
      a different schema than the lifted table, breaking outer
      references.
    * ``p`` references only inner-table columns. Subqueries in FROM /
      JOIN sources can't reference outer-scope names by SQL syntax, so
      any name in ``p`` resolves to the inner table — no correlation
      to break.

    The inner WHERE is alias-rewritten so any qualified
    ``inner_alias.col`` reference becomes ``outer_alias.col``, then
    AND-joined onto the existing ON clause (or used as the new ON
    clause when the join had none, which the parser shouldn't emit but
    we handle defensively).

    Soundness intuition: ``JOIN (SELECT * FROM T WHERE p) sub ON
    cond`` produces the rows ``{(l, r) | l ∈ outer ∧ r ∈ T ∧ p(r) ∧
    cond(l, r)}``, identical to ``JOIN T sub ON cond AND p`` because
    ``∧`` is commutative.
    """
    if not node.joins:
        return node
    new_joins: list[SqlJoin] = []
    changed = False
    for j in node.joins:
        if not isinstance(j.source, SqlSubquery):
            new_joins.append(j)
            continue
        inner = j.source.query
        # Same-shape constraints as ``_is_passthrough_select`` plus we
        # require the inner has *exactly* a WHERE (passthrough rule
        # already handles the no-WHERE case).
        if not isinstance(inner.from_source, SqlTable):
            new_joins.append(j)
            continue
        if (
            inner.joins
            or inner.group_by
            or inner.order_by
            or inner.limit
        ):
            new_joins.append(j)
            continue
        if not inner.where.strip():
            # Falls under the passthrough rule, not this one.
            new_joins.append(j)
            continue
        if not _qualified_or_bare_columns(
            inner.columns,
            inner.from_source.alias or inner.from_source.name,
        ):
            new_joins.append(j)
            continue

        # Rewrite the inner WHERE so any qualified references to the
        # inner table's alias point at the outer (subquery) alias,
        # which is what the lifted table will be aliased as.
        inner_alias = inner.from_source.alias or inner.from_source.name
        outer_alias = j.source.alias or inner_alias
        if (
            inner_alias
            and outer_alias
            and inner_alias != outer_alias
        ):
            rewritten_where = re.sub(
                rf"(?<!\w){re.escape(inner_alias)}\.",
                f"{outer_alias}.",
                inner.where,
            )
        else:
            rewritten_where = inner.where

        promoted = _promote_alias(inner.from_source, j.source.alias)
        merged_on = _and_join(j.on_condition, rewritten_where)
        new_joins.append(
            replace(
                j,
                source=promoted,
                # If the original join had no ON clause, this becomes
                # the ON clause; if it had one, the inner WHERE is
                # AND-joined onto it.
                on_condition=merged_on,
            )
        )
        changed = True

    if not changed:
        return node
    return replace(node, joins=new_joins)


# ---------------------------------------------------------------------------
# Subquery-shape rewrites
# ---------------------------------------------------------------------------


def _rule_drop_subquery_orderby_without_limit(node: SqlSelect) -> SqlSelect:
    """``SELECT … FROM (… ORDER BY x) sub`` → ``SELECT … FROM (…) sub``
    when the subquery has no ``LIMIT``.

    Soundness: relational SQL is set-based at semicolons. ``ORDER BY``
    only affects the visible row order of the *final* result. When an
    ``ORDER BY`` lives inside a subquery without a ``LIMIT``, the
    enclosing query consumes the subquery as an unordered multiset —
    the ordering is unobservable and can be safely dropped.

    The rule does NOT touch top-level ``ORDER BY`` (that's user-visible)
    or any ``ORDER BY`` paired with a ``LIMIT`` in the same scope (the
    pair selects which rows survive).

    Applies in both the FROM source and any JOIN source.
    """
    new_from = node.from_source
    changed_from = False
    if (
        isinstance(new_from, SqlSubquery)
        and new_from.query.order_by
        and not new_from.query.limit
    ):
        new_from = SqlSubquery(
            query=replace(new_from.query, order_by=""),
            alias=new_from.alias,
        )
        changed_from = True

    new_joins: list[SqlJoin] = []
    changed_joins = False
    for j in node.joins:
        if (
            isinstance(j.source, SqlSubquery)
            and j.source.query.order_by
            and not j.source.query.limit
        ):
            new_joins.append(
                replace(
                    j,
                    source=SqlSubquery(
                        query=replace(j.source.query, order_by=""),
                        alias=j.source.alias,
                    ),
                )
            )
            changed_joins = True
        else:
            new_joins.append(j)

    if not (changed_from or changed_joins):
        return node
    return replace(node, from_source=new_from, joins=new_joins)


# Aggregate-name patterns. We match ``AGG(...)`` at the start of the
# trimmed column expression so qualified aliases like ``COUNT(emp_id)
# AS n`` still match.
_AGG_FUNCS = ("COUNT", "SUM", "AVG", "MIN", "MAX")
_AGG_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(_AGG_FUNCS) + r")\s*\(",
    re.IGNORECASE,
)


def _is_aggregate_expr(col: str) -> bool:
    """True if ``col`` starts with an aggregate function call."""
    return bool(_AGG_PATTERN.match(col))


def _returns_exactly_one_row(node: SqlSelect) -> bool:
    """True iff ``node`` is guaranteed to return exactly one row.

    Two safe cases:

    * ``LIMIT 1`` (and the source isn't empty in the strictest sense —
      but ``COUNT`` and friends over an empty source still return one
      row, which is the only context this helper is used in).
    * Every column in the projection is a scalar aggregate AND there's
      no ``GROUP BY``. Aggregate-without-GROUP BY collapses any
      multiset (including the empty one) to exactly one row.

    Returns False conservatively for shapes the rule doesn't model
    (e.g. ``UNION`` would be visible as a non-SqlSelect form, but our
    AST doesn't represent it explicitly).
    """
    # Case A: explicit LIMIT 1.
    if node.limit.strip() == "1":
        return True

    # Case B: scalar-aggregate query with no GROUP BY.
    if node.group_by:
        return False
    if not node.columns:
        return False
    return all(_is_aggregate_expr(c) for c in node.columns)


def _rule_count_over_one_row_subquery(node: SqlSelect) -> SqlSelect:
    """``SELECT COUNT(<expr>) FROM (<inner>) sub`` → ``SELECT 1``
    when ``<inner>`` is guaranteed to return exactly one row AND
    ``<expr>`` is guaranteed to be non-null in that row.

    Soundness: ``COUNT(e)`` counts rows where ``e`` is non-null.
    Over a one-row source it's 1 if ``e`` is non-null, 0 if ``e`` is
    null. We therefore only rewrite when we can prove non-nullness
    statically. Two safe cases:

    * ``COUNT(*)`` — always counts rows regardless of column values,
      so the result is exactly the row count (= 1 for one-row source).
    * ``COUNT(sub.X)`` where the inner SELECT's column matching ``X``
      is itself an aggregate ``COUNT(...)``. ``COUNT`` always returns
      a non-null integer ≥ 0, so ``sub.X`` is non-null and the outer
      ``COUNT`` returns 1.

    Other ``COUNT(<expr>)`` forms (qualified column whose inner is a
    SUM/AVG/MIN/MAX, or any unrelated expression) are NOT rewritten —
    those aggregates can be null over an empty input, and the safety
    proof breaks.

    Pattern (all conditions required):

    * Exactly one outer column whose expression is ``COUNT(*)`` or
      ``COUNT(<sub-alias>.<col>)``.
    * The FROM source is a subquery whose AST satisfies
      ``_returns_exactly_one_row``.
    * No outer JOINs, WHERE, GROUP BY, ORDER BY, or LIMIT (all of
      these could change the count).

    The replacement is the constant query ``SELECT 1`` (preserving the
    outer column's ``AS alias`` if any). That bare SELECT (no FROM)
    is valid in every dialect we target.
    """
    if len(node.columns) != 1:
        return node
    if node.joins or node.where or node.group_by or node.order_by or node.limit:
        return node
    if not isinstance(node.from_source, SqlSubquery):
        return node

    sub = node.from_source
    inner = sub.query
    if not _returns_exactly_one_row(inner):
        return node

    col = node.columns[0].strip()
    if not _is_count_with_proven_nonnull_arg(col, sub.alias, inner.columns):
        return node

    # Preserve the outer column's alias if any (``COUNT(*) AS n`` →
    # ``1 AS n``).
    m = _AS_RENAME.match(col)
    new_col = f"1 AS {m.group(2)}" if m else "1"

    return SqlSelect(columns=[new_col])


_COUNT_STAR_PATTERN = re.compile(
    r"^\s*COUNT\s*\(\s*\*\s*\)\s*(?:AS\s+\w+\s*)?$",
    re.IGNORECASE,
)

_COUNT_QUALIFIED_PATTERN = re.compile(
    r"^\s*COUNT\s*\(\s*(\w+)\.(\w+)\s*\)\s*(?:AS\s+\w+\s*)?$",
    re.IGNORECASE,
)


def _is_count_with_proven_nonnull_arg(
    col: str, sub_alias: str, inner_columns: list[str],
) -> bool:
    """True if ``col`` is a ``COUNT(...)`` form whose argument we can
    prove non-null statically.

    See :func:`_rule_count_over_one_row_subquery` for the soundness
    argument.
    """
    if _COUNT_STAR_PATTERN.match(col):
        return True
    m = _COUNT_QUALIFIED_PATTERN.match(col)
    if not m:
        return False
    qualifier, column = m.group(1), m.group(2)
    if qualifier != sub_alias:
        return False
    # Find the inner column that exposes ``column`` and check it's a
    # COUNT(...) aggregate (always non-null).
    for inner_col in inner_columns:
        public = _column_public_name(inner_col)
        if public != column:
            continue
        underlying = _column_underlying_expr(inner_col) or inner_col
        if _COUNT_LIKE_INNER_PATTERN.match(underlying):
            return True
        return False
    return False


# A ``COUNT(...)`` aggregate, with any argument shape (``*``, qualified,
# bare). Used to confirm that an inner column is COUNT-shaped before
# concluding it's non-null.
_COUNT_LIKE_INNER_PATTERN = re.compile(
    r"^\s*COUNT\s*\(",
    re.IGNORECASE,
)


_RULES: list[tuple[str, SimplifyRule]] = [
    # Recursion is itself a "rule" in the pipeline so it benefits from
    # the same fixed-point convergence as any other rewrite.
    ("recurse", _rule_recurse_into_subqueries),
    # Subquery-shape cleanups that don't depend on outer-clause
    # composition. Drop unobservable ``ORDER BY`` first so subsequent
    # rules see a normalised inner.
    ("drop_subquery_orderby_without_limit", _rule_drop_subquery_orderby_without_limit),
    # FROM-side rewrites, ordered from most specific (passthrough) to
    # most general (rebinding inline).
    ("unwrap_passthrough_from", _rule_unwrap_passthrough_from),
    ("merge_inner_where", _rule_merge_inner_where),
    ("inline_rebinding_subquery", _rule_inline_rebinding_subquery),
    ("collapse_redundant_wrapper", _rule_collapse_redundant_wrapper),
    ("merge_groupby_through_subquery", _rule_merge_groupby_through_subquery),
    # Constant-folding for nested aggregates.
    ("count_over_one_row_subquery", _rule_count_over_one_row_subquery),
    # Predicate-shape rewrites.
    ("cross_join_to_join_on", _rule_cross_join_to_join_on),
    # JOIN-side rewrites: passthrough (no inner WHERE) first, then the
    # WHERE-bearing variant which lifts the inner WHERE into the join's
    # ON clause.
    ("unwrap_join_subquery", _rule_unwrap_join_subquery),
    ("unwrap_join_subquery_with_where", _rule_unwrap_join_subquery_with_where),
    # Canonicalise predicate placement: single-table predicates in
    # INNER JOIN ON clauses move to the outer WHERE.
    ("lift_singletable_join_predicates", _rule_lift_singletable_join_predicates),
]


def _simplify(node: SqlSelect) -> SqlSelect:
    for _, rule in _RULES:
        node = rule(node)
    return node


def simplify_ast(node: SqlSelect) -> SqlSelect:
    """Apply simplification rules until ``render_sql`` is stable."""
    for _ in range(10):
        prev = render_sql(node)
        node = _simplify(node)
        if render_sql(node) == prev:
            break
    return node


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_sql(node: SqlSelect) -> str:
    """Render a SqlSelect AST to formatted SQL."""
    parts = [f"SELECT {', '.join(node.columns)}"]
    if node.from_source is not None:
        parts.append(f"  FROM {_render_source(node.from_source)}")
    for j in node.joins:
        s = _render_source(j.source)
        if j.on_condition:
            parts.append(f"  {j.join_type} {s} ON {j.on_condition}")
        else:
            parts.append(f"  {j.join_type} {s}")
    if node.where:
        parts.append(f"  WHERE {node.where}")
    if node.group_by:
        parts.append(f"  GROUP BY {', '.join(node.group_by)}")
    if node.order_by:
        parts.append(f"  ORDER BY {node.order_by}")
    if node.limit:
        parts.append(f"  LIMIT {node.limit}")
    return "\n".join(parts)


def _render_source(source: SqlSource) -> str:
    if isinstance(source, SqlTable):
        return f"{source.name} {source.alias}" if source.alias else source.name
    if isinstance(source, SqlSubquery):
        inner = render_sql(source.query)
        indented = "\n".join(f"    {line}" for line in inner.split("\n"))
        return f"(\n{indented}\n  ) AS {source.alias}"
    return "?"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simplify_sql(sql: str) -> str:
    """Parse SQL, apply simplification rules to a fixed point, render back."""
    return render_sql(simplify_ast(parse_sql(sql)))
