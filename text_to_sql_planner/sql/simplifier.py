"""SQL simplifier: applies correctness-preserving transformations via a parse tree.

Parses SQL into an AST, applies simplification rules, then renders back to SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union
import re


# --- SQL AST ---

@dataclass
class SqlTable:
    """A plain table reference."""
    name: str


@dataclass
class SqlSubquery:
    """A subquery used as a FROM source."""
    query: SqlSelect
    alias: str = "sub"


SqlSource = Union[SqlTable, SqlSubquery]


@dataclass
class SqlJoin:
    """A JOIN clause."""
    join_type: str
    source: SqlSource
    on_condition: str = ""


@dataclass
class SqlSelect:
    """A SELECT statement."""
    columns: list[str] = field(default_factory=list)
    from_source: SqlSource | None = None
    joins: list[SqlJoin] = field(default_factory=list)
    where: str = ""
    group_by: list[str] = field(default_factory=list)


# --- Parser ---

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
            end = _find_first_keyword(rest, ["JOIN", "CROSS JOIN", "WHERE", "GROUP BY"])
            if end == -1:
                on_cond = rest.strip()
                rest = ""
            else:
                on_cond = rest[:end].strip()
                rest = rest[end:]
        joins.append(SqlJoin(join_type=jtype, source=jsrc, on_condition=on_cond))

    where = ""
    rest = rest.strip()
    wm = re.match(r"WHERE\s+", rest, re.IGNORECASE)
    if wm:
        rest = rest[wm.end():]
        gb = _find_keyword(rest, "GROUP BY")
        if gb == -1:
            where = rest.strip()
            rest = ""
        else:
            where = rest[:gb].strip()
            rest = rest[gb:]

    group_by: list[str] = []
    rest = rest.strip()
    gm = re.match(r"GROUP\s+BY\s+", rest, re.IGNORECASE)
    if gm:
        group_by = [c.strip() for c in rest[gm.end():].split(",")]

    return SqlSelect(columns=columns, from_source=from_source, joins=joins, where=where, group_by=group_by)


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
        return SqlSubquery(query=_parse_select(inner), alias=alias), rest
    else:
        m = re.match(r"(\w+)", text)
        if m:
            return SqlTable(name=m.group(1)), text[m.end():]
        return SqlTable(name=text), ""


def _find_keyword(text: str, keyword: str) -> int:
    depth = 0
    kl = len(keyword)
    tu = text.upper()
    ku = keyword.upper()
    for i in range(len(text) - kl + 1):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
        elif depth == 0 and tu[i:i+kl] == ku:
            if (i == 0 or not text[i-1].isalnum()) and (i+kl >= len(text) or not text[i+kl].isalnum()):
                return i
    return -1


def _find_first_keyword(text: str, keywords: list[str]) -> int:
    positions = [p for p in (_find_keyword(text, kw) for kw in keywords) if p != -1]
    return min(positions) if positions else -1


def _find_matching_paren(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


# --- Simplification ---

def simplify_ast(node: SqlSelect) -> SqlSelect:
    """Apply simplification rules recursively until fixpoint."""
    for _ in range(10):
        prev = render_sql(node)
        node = _simplify_once(node)
        if render_sql(node) == prev:
            break
    return node


def _simplify_once(node: SqlSelect) -> SqlSelect:
    # Recursively simplify subqueries first
    fs = node.from_source
    if isinstance(fs, SqlSubquery):
        fs = SqlSubquery(query=_simplify_once(fs.query), alias=fs.alias)

    joins = []
    for j in node.joins:
        src = j.source
        if isinstance(src, SqlSubquery):
            src = SqlSubquery(query=_simplify_once(src.query), alias=src.alias)
        joins.append(SqlJoin(join_type=j.join_type, source=src, on_condition=j.on_condition))

    node = SqlSelect(columns=node.columns, from_source=fs, joins=joins, where=node.where, group_by=node.group_by)

    # Rule: unwrap trivial subquery in FROM
    # SELECT cols FROM (SELECT cols2 FROM T) AS sub → SELECT cols FROM T
    # Condition: inner has no WHERE, no GROUP BY, no JOINs, from_source is a table
    if isinstance(node.from_source, SqlSubquery):
        inner = node.from_source.query
        if (isinstance(inner.from_source, SqlTable)
                and not inner.joins
                and not inner.where
                and not inner.group_by):
            # Safe to unwrap
            node = SqlSelect(
                columns=node.columns,
                from_source=inner.from_source,
                joins=node.joins,
                where=node.where,
                group_by=node.group_by,
            )

    # Rule: merge inner WHERE when unwrapping
    if isinstance(node.from_source, SqlSubquery):
        inner = node.from_source.query
        if (isinstance(inner.from_source, SqlTable)
                and not inner.joins
                and not inner.group_by):
            # Can merge: pull inner WHERE into outer
            merged_where = inner.where
            if node.where and merged_where:
                merged_where = f"{merged_where} AND {node.where}"
            elif node.where:
                merged_where = node.where
            node = SqlSelect(
                columns=node.columns,
                from_source=inner.from_source,
                joins=node.joins,
                where=merged_where,
                group_by=node.group_by,
            )

    # Rule: collapse redundant wrapper
    # SELECT cols FROM (SELECT same_cols FROM ... WHERE ... GROUP BY ...) AS sub
    # → inner query (when outer adds nothing: no WHERE, no JOINs, no GROUP BY)
    if (isinstance(node.from_source, SqlSubquery)
            and not node.joins
            and not node.where
            and not node.group_by):
        inner = node.from_source.query
        outer_norm = _norm_cols(node.columns)
        inner_norm = _norm_cols(inner.columns)
        if outer_norm == inner_norm:
            # Outer is a pure passthrough — replace with inner
            node = inner

    return node


# --- Renderer ---

def render_sql(node: SqlSelect) -> str:
    """Render a SqlSelect AST to formatted SQL."""
    cols = ", ".join(node.columns)
    parts = [f"SELECT {cols}"]
    if node.from_source:
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
    return "\n".join(parts)


def _render_source(source: SqlSource) -> str:
    if isinstance(source, SqlTable):
        return source.name
    elif isinstance(source, SqlSubquery):
        inner = render_sql(source.query)
        indented = "\n".join(f"    {line}" for line in inner.split("\n"))
        return f"(\n{indented}\n  ) AS {source.alias}"
    return "?"


# --- Public API ---

def simplify_sql(sql: str) -> str:
    """Parse SQL, simplify, and render back."""
    ast = parse_sql(sql)
    simplified = simplify_ast(ast)
    return render_sql(simplified)
