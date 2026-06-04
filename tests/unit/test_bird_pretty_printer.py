"""Smoke tests for ``bird_benchmark.sql_to_drc.pretty_printer.drc_to_sql``.

These tests exercise the round-trip pretty-printer shim (task 15.1) on a
small set of representative SELECT statements. They are not the round-
trip property test (task 15.2) — that test will run cvc5 over many
generated inputs. Here we just verify two contracts:

1. For a representative SELECT in the Supported_SQL_Subset, the shim
   produces a SQL string that re-parses cleanly through
   :func:`bird_benchmark.sql_to_drc.convert_sql`.
2. Failure cases (empty DRC) surface a structured
   :class:`bird_benchmark.types.ConverterError` rather than raising.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import DRCExpression

from bird_benchmark.sql_to_drc import convert_sql
from bird_benchmark.sql_to_drc.pretty_printer import drc_to_sql
from bird_benchmark.types import ConverterError


_SCHEMA_PEOPLE = (
    "CREATE TABLE people (id INTEGER, name TEXT, age INTEGER);\n"
    "CREATE TABLE pets (id INTEGER, owner_id INTEGER, kind TEXT);"
)


def _convert_or_fail(sql: str, schema: str):
    result = convert_sql(sql, schema)
    assert not isinstance(result, ConverterError), (
        f"convert_sql({sql!r}) failed: {result}"
    )
    return result


def test_drc_to_sql_round_trips_simple_select_to_parseable_sql():
    """SELECT * FROM people → drc1 → sql_mid → drc2 (no error).

    The shim uses the DRC's bound-variable names (e.g.
    ``v_people_age_3``) as column names in the rendered SQL — it has
    no way to recover the original column names from the DRC alone.
    Re-parsing therefore uses the translator's no-schema fallback
    (``_collect_referenced_columns``), which discovers the synthetic
    names from the SQL body and binds them to fresh DRC variables.
    The round-trip property test (task 15.2) exploits this same path.
    """
    sql_in = "SELECT id, name FROM people WHERE age > 18"
    drc1 = _convert_or_fail(sql_in, _SCHEMA_PEOPLE)

    sql_mid = drc_to_sql(drc1)
    assert isinstance(sql_mid, str), (
        f"expected SQL string, got ConverterError: {sql_mid}"
    )
    assert "people" in sql_mid
    assert "SELECT" in sql_mid.upper()

    # Re-parse with an empty schema so the translator's fallback
    # path picks up the synthetic variable names from the body.
    drc2 = convert_sql(sql_mid, "")
    assert not isinstance(drc2, ConverterError), (
        f"sql_mid did not re-parse: {drc2}\nsql_mid was:\n{sql_mid}"
    )


def test_drc_to_sql_round_trips_join_select():
    """A two-table inner join produces well-formed SQL.

    The shim's contract is to return a SQL string; the round-trip
    property test (task 15.2) is the place that exercises end-to-end
    semantic equivalence via cvc5. The planner's SQL converter
    happens to render the cartesian-product + selection shape as
    ``CROSS JOIN ... WHERE ...``, which the bird_benchmark parser
    does not yet accept on the re-parse step. Property 13 explicitly
    records that artifact-level failure case, so we don't assert
    re-parsability here — only that the shim stays in the success
    path and emits both relation names.
    """
    sql_in = (
        "SELECT name, kind FROM people "
        "INNER JOIN pets ON people.id = pets.owner_id"
    )
    drc1 = _convert_or_fail(sql_in, _SCHEMA_PEOPLE)

    sql_mid = drc_to_sql(drc1)
    assert isinstance(sql_mid, str), (
        f"expected SQL string, got ConverterError: {sql_mid}"
    )
    assert "people" in sql_mid
    assert "pets" in sql_mid
    assert "SELECT" in sql_mid.upper()


def test_drc_to_sql_returns_converter_error_for_empty_drc():
    """A DRC with no MembershipNodes yields a structured ConverterError.

    The shim must never raise — every failure flows through the
    ``ConverterError`` return channel so the round-trip harness can
    record it as one of the four artifacts.
    """
    empty = DRCExpression(result_variables=[], condition=None)

    result = drc_to_sql(empty)
    assert isinstance(result, ConverterError)
    assert result.kind == "parse_error"
    assert "FROM" in result.message or "Membership" in result.message
