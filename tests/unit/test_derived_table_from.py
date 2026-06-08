"""Tests for derived-table FROM source support in the SQL→DRC pipeline.

Background — run-11 dev_197:
BIRD's gold SQL was

    SELECT AVG(oxygen_count)
    FROM (SELECT T1.molecule_id, COUNT(T1.element) AS oxygen_count
          FROM atom AS T1 INNER JOIN bond AS T2 ON …
          WHERE T2.bond_type = '-' AND T1.element = 'o'
          GROUP BY T1.molecule_id) AS oxygen_counts

The parser previously rejected the parenthesised SELECT after FROM
with ``parse_error: expected identifier, got '('``, so the test case
came back ``skipped``. This test file pins the parser / translator
support for derived-table FROM sources end-to-end:

* The parser accepts ``FROM (SELECT …) AS alias``.
* The translator inlines the subquery's body into the outer DRC.
* The outer query can reference the subquery's projected columns
  by their alias-qualified name.
* Aliases on aggregate select-list items (``COUNT(x) AS n``)
  expose ``alias.n`` as an outer-scope column.
"""

from __future__ import annotations

import pytest

from bird_benchmark.sql_to_drc import convert_sql
from bird_benchmark.types import ConverterError
from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ColumnVariable,
    LimitExpression,
    OrderByExpression,
    QuantifierNode,
)


def _strip_wrappers(query):
    while isinstance(query, (LimitExpression, OrderByExpression)):
        query = query.inner
    return query


_SCHEMA_TOXICOLOGY = (
    "CREATE TABLE atom ("
    "  atom_id TEXT NOT NULL,"
    "  molecule_id TEXT,"
    "  element TEXT,"
    "  PRIMARY KEY (atom_id)"
    ");\n"
    "CREATE TABLE bond ("
    "  bond_id TEXT NOT NULL,"
    "  molecule_id TEXT,"
    "  bond_type TEXT,"
    "  PRIMARY KEY (bond_id)"
    ");"
)


def test_parser_accepts_derived_table_from_source():
    """``FROM (SELECT col FROM t) AS alias`` no longer errors."""
    sql = "SELECT a.x FROM (SELECT id AS x FROM atom) AS a"
    schema = "CREATE TABLE atom (id INTEGER);"
    result = convert_sql(sql, schema)
    assert not isinstance(result, ConverterError), (
        f"expected success, got {result}"
    )


def test_parser_requires_alias_on_derived_table():
    """SQLite enforces aliasing on derived tables; we do too."""
    sql = "SELECT * FROM (SELECT id FROM atom)"
    schema = "CREATE TABLE atom (id INTEGER);"
    result = convert_sql(sql, schema)
    assert isinstance(result, ConverterError)
    assert "alias" in result.message.lower()


def test_derived_table_exposes_inner_select_alias_as_outer_column():
    """When the inner SELECT uses ``COUNT(x) AS n``, the outer query
    can reference ``alias.n``. This is the dev_197 shape."""
    sql = (
        "SELECT a.oxygen_count "
        "FROM (SELECT molecule_id, COUNT(element) AS oxygen_count "
        "      FROM atom GROUP BY molecule_id) AS a"
    )
    result = convert_sql(sql, _SCHEMA_TOXICOLOGY)
    assert not isinstance(result, ConverterError), (
        f"expected success, got {result}"
    )

    drc = _strip_wrappers(result)
    # The result variable corresponds to ``a.oxygen_count`` — its name
    # is the inner alias's bound variable, exposed via the outer scope.
    assert len(drc.result_variables) == 1


def test_dev_197_derived_table_with_outer_avg():
    """The full dev_197 gold SQL: outer ``AVG(oxygen_count)`` over a
    derived table that does ``GROUP BY molecule_id``."""
    sql = (
        "SELECT AVG(oxygen_count) "
        "FROM (SELECT T1.molecule_id, COUNT(T1.element) AS oxygen_count "
        "      FROM atom AS T1 INNER JOIN bond AS T2 "
        "      ON T1.molecule_id = T2.molecule_id "
        "      WHERE T2.bond_type = '-' AND T1.element = 'o' "
        "      GROUP BY T1.molecule_id) AS oxygen_counts"
    )
    result = convert_sql(sql, _SCHEMA_TOXICOLOGY)
    assert not isinstance(result, ConverterError), (
        f"expected success, got {result}"
    )

    drc = _strip_wrappers(result)
    # Outer SELECT projects exactly one aggregate.
    assert len(drc.result_variables) == 1
    rv = drc.result_variables[0]
    assert isinstance(rv, AggregateVariable)
    assert rv.function == "AVG"


def test_derived_table_with_join_to_real_table():
    """A derived table can be joined to a real table — the outer scope
    sees both the derived-table aliases AND the real-table columns."""
    sql = (
        "SELECT a.x, b.element "
        "FROM (SELECT id AS x FROM atom) AS a "
        "INNER JOIN bond AS b ON a.x = b.bond_id"
    )
    schema = (
        "CREATE TABLE atom (id INTEGER);\n"
        "CREATE TABLE bond (bond_id TEXT, element TEXT);"
    )
    result = convert_sql(sql, schema)
    assert not isinstance(result, ConverterError), (
        f"expected success, got {result}"
    )


def test_derived_table_inner_filter_survives_to_outer_drc():
    """The inner SELECT's WHERE filters must reach the outer DRC's
    condition — otherwise ``SELECT … FROM (SELECT … WHERE …)`` would
    silently lose the filter."""
    sql = (
        "SELECT a.element "
        "FROM (SELECT element FROM atom WHERE element = 'o') AS a"
    )
    schema = "CREATE TABLE atom (element TEXT);"
    result = convert_sql(sql, schema)
    assert not isinstance(result, ConverterError)

    drc = _strip_wrappers(result)
    # Render the condition to text and confirm the literal "o" appears
    # somewhere — that's the trace of the inner WHERE.
    cond_text = repr(drc.condition)
    assert '"o"' in cond_text or "'o'" in cond_text or "value='o'" in cond_text
