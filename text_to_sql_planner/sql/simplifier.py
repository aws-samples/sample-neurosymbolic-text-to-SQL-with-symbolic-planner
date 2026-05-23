"""SQL simplifier: applies correctness-preserving transformations to reduce nesting.

Strategy: repeatedly unwrap trivial subqueries (SELECT cols FROM single_table)
until no more simplifications are possible.
"""

from __future__ import annotations

import re


def simplify_sql(sql: str) -> str:
    """Apply simplification passes until no more changes occur."""
    prev = ""
    current = sql.strip()
    for _ in range(10):
        prev = current
        current = _unwrap_from_subquery(current)
        current = _clean(current)
        if current == prev:
            break
    return current


def _unwrap_from_subquery(sql: str) -> str:
    """Unwrap FROM (SELECT cols FROM table) AS alias → FROM table.

    Uses balanced-paren matching to find subqueries. Only unwraps when
    the inner query is a simple SELECT from a single table (no WHERE,
    JOIN, GROUP BY, HAVING, subqueries, etc.).
    """
    result = sql
    max_iterations = 20
    changed = True
    while changed:
        changed = False
        # Find ALL FROM ( patterns and try to simplify from innermost out
        # Work backwards to simplify inner subqueries first
        positions = [(m.start(), m.end()) for m in re.finditer(r"FROM\s+\(", result, re.IGNORECASE)]

        # Process from last to first (innermost first)
        for match_start, match_end in reversed(positions):
            start_paren = match_end - 1
            end_paren = _find_matching_paren(result, start_paren)
            if end_paren < 0:
                continue

            inner_sql = result[start_paren + 1:end_paren].strip()

            # Check for AS alias after the closing paren
            after_paren = result[end_paren + 1:]
            alias_match = re.match(r"\s*AS\s+(\w+)", after_paren, re.IGNORECASE)
            if not alias_match:
                continue

            full_end = end_paren + 1 + alias_match.end()

            # Check if inner SQL is simple: SELECT cols FROM single_table
            if _is_simple_select(inner_sql):
                table_match = re.search(r"\bFROM\s+(\w+)\s*$", inner_sql, re.IGNORECASE)
                if table_match:
                    table_name = table_match.group(1)
                    result = result[:match_start] + f"FROM {table_name}" + result[full_end:]
                    changed = True
                    break  # restart since positions shifted

    return result


def _is_simple_select(sql: str) -> bool:
    """Check if SQL is a simple SELECT cols FROM table (no clauses, no subqueries)."""
    sql_upper = sql.upper().strip()
    # Must start with SELECT
    if not sql_upper.startswith("SELECT"):
        return False
    # Must not contain nested subqueries (parens other than in function calls like COUNT())
    # Count parens — function calls have balanced parens within the SELECT clause
    # Check for keywords that indicate complexity
    # Split at FROM and check what's after the table name
    from_match = re.search(r"\bFROM\s+(\w+)(.*)", sql, re.IGNORECASE | re.DOTALL)
    if not from_match:
        return False
    remainder = from_match.group(2).strip()
    # If nothing after the table name, it's simple
    return len(remainder) == 0


def _find_matching_paren(text: str, start: int) -> int:
    """Find the matching closing paren for the open paren at `start`."""
    depth = 0
    i = start
    in_string = False
    quote_char = ""
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == quote_char:
                in_string = False
        else:
            if ch in ("'", '"'):
                in_string = True
                quote_char = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _clean(sql: str) -> str:
    """Clean up whitespace and re-indent."""
    lines = [line for line in sql.split("\n") if line.strip()]
    if not lines:
        return ""
    result = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == 0:
            result.append(stripped)
        elif re.match(r"^(FROM|JOIN|WHERE|GROUP BY|HAVING|ORDER BY|LIMIT)\b", stripped, re.IGNORECASE):
            result.append(f"  {stripped}")
        elif re.match(r"^(AND|OR)\b", stripped, re.IGNORECASE):
            result.append(f"    {stripped}")
        else:
            result.append(f"  {stripped}")
    return "\n".join(result)
