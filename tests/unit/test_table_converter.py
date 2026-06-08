"""Unit tests for the table converter."""

import pytest

from text_to_sql_planner.converter.table_converter import (
    convert_tables,
    TableRelation,
    TableConversionSuccess,
    TableConversionFailure,
)
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    MembershipNode,
)


class TestSingleTable:
    """Test conversion of a single CREATE TABLE statement."""

    def test_simple_table_with_multiple_columns(self):
        schema = "CREATE TABLE users (id INT, name VARCHAR(100), email TEXT);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert len(result.relations) == 1

        rel = result.relations[0]
        assert rel.table_name == "users"
        assert rel.columns == ["id", "name", "email"]

    def test_expression_has_correct_result_variables(self):
        schema = "CREATE TABLE orders (order_id INT, amount DECIMAL(10,2));"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        rel = result.relations[0]

        assert len(rel.expression.result_variables) == 2
        assert rel.expression.result_variables[0] == ColumnVariable(name="order_id")
        assert rel.expression.result_variables[1] == ColumnVariable(name="amount")

    def test_expression_has_membership_condition(self):
        schema = "CREATE TABLE products (id INT, name TEXT, price DECIMAL);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        rel = result.relations[0]

        condition = rel.expression.condition
        assert isinstance(condition, MembershipNode)
        assert condition.variables == ["id", "name", "price"]
        assert condition.relation == "products"


class TestMultipleTables:
    """Test conversion of multiple CREATE TABLE statements."""

    def test_two_tables(self):
        schema = """
        CREATE TABLE customers (id INT, name VARCHAR(50));
        CREATE TABLE orders (order_id INT, customer_id INT, total DECIMAL);
        """
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert len(result.relations) == 2

        assert result.relations[0].table_name == "customers"
        assert result.relations[0].columns == ["id", "name"]

        assert result.relations[1].table_name == "orders"
        assert result.relations[1].columns == ["order_id", "customer_id", "total"]

    def test_three_tables(self):
        schema = """
        CREATE TABLE a (x INT);
        CREATE TABLE b (y TEXT, z INT);
        CREATE TABLE c (p VARCHAR(10), q DECIMAL(5,2), r BOOLEAN);
        """
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert len(result.relations) == 3
        assert result.relations[0].table_name == "a"
        assert result.relations[1].table_name == "b"
        assert result.relations[2].table_name == "c"
        assert result.relations[2].columns == ["p", "q", "r"]


class TestSQLTypes:
    """Test handling of various SQL column types."""

    def test_int_types(self):
        schema = "CREATE TABLE t (a INT, b INTEGER, c BIGINT, d SMALLINT);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["a", "b", "c", "d"]

    def test_varchar_with_length(self):
        schema = "CREATE TABLE t (name VARCHAR(255), code CHAR(10));"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["name", "code"]

    def test_text_types(self):
        schema = "CREATE TABLE t (bio TEXT, notes CLOB, data BLOB);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["bio", "notes", "data"]

    def test_decimal_with_precision_and_scale(self):
        schema = "CREATE TABLE t (price DECIMAL(10,2), rate NUMERIC(5,4));"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["price", "rate"]

    def test_date_and_time_types(self):
        schema = "CREATE TABLE t (created DATE, updated TIMESTAMP, duration TIME);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["created", "updated", "duration"]

    def test_boolean_type(self):
        schema = "CREATE TABLE t (active BOOLEAN, verified BOOL);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["active", "verified"]


class TestConstraints:
    """Test handling of column and table-level constraints."""

    def test_primary_key_inline(self):
        schema = "CREATE TABLE t (id INT PRIMARY KEY, name TEXT);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["id", "name"]

    def test_not_null_constraint(self):
        schema = "CREATE TABLE t (id INT NOT NULL, name VARCHAR(50) NOT NULL);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["id", "name"]

    def test_default_value(self):
        schema = "CREATE TABLE t (id INT, status VARCHAR(20) DEFAULT 'active');"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["id", "status"]

    def test_table_level_primary_key_constraint(self):
        schema = """CREATE TABLE t (
            id INT,
            name TEXT,
            PRIMARY KEY (id)
        );"""
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        # PRIMARY KEY (id) is a table-level constraint, should be skipped
        assert result.relations[0].columns == ["id", "name"]

    def test_table_level_foreign_key_constraint(self):
        schema = """CREATE TABLE orders (
            id INT,
            customer_id INT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );"""
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["id", "customer_id"]

    def test_unique_constraint(self):
        schema = "CREATE TABLE t (id INT, email VARCHAR(100) UNIQUE);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["id", "email"]

    def test_multiple_constraints_combined(self):
        schema = """CREATE TABLE employees (
            id INT PRIMARY KEY NOT NULL,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(200) UNIQUE NOT NULL,
            dept_id INT,
            FOREIGN KEY (dept_id) REFERENCES departments(id)
        );"""
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["id", "name", "email", "dept_id"]


class TestErrorCases:
    """Test error handling."""

    def test_empty_string(self):
        result = convert_tables("")
        assert isinstance(result, TableConversionFailure)
        assert "Empty schema" in result.error

    def test_whitespace_only(self):
        result = convert_tables("   \n\t  ")
        assert isinstance(result, TableConversionFailure)
        assert "Empty schema" in result.error

    def test_unparseable_input(self):
        result = convert_tables("SELECT * FROM users;")
        assert isinstance(result, TableConversionFailure)
        assert "No valid CREATE TABLE" in result.error

    def test_random_text(self):
        result = convert_tables("this is not SQL at all")
        assert isinstance(result, TableConversionFailure)
        assert "No valid CREATE TABLE" in result.error

    def test_table_with_only_constraints_no_columns(self):
        schema = """CREATE TABLE t (
            PRIMARY KEY (id),
            UNIQUE (name)
        );"""
        result = convert_tables(schema)
        assert isinstance(result, TableConversionFailure)
        assert "no column definitions" in result.error


class TestColumnOrderPreservation:
    """Test that column order is preserved from the definition."""

    def test_order_matches_definition(self):
        schema = "CREATE TABLE t (zebra INT, alpha TEXT, middle VARCHAR(50));"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        rel = result.relations[0]
        assert rel.columns == ["zebra", "alpha", "middle"]

        # Result variables should also be in definition order
        names = [rv.name for rv in rel.expression.result_variables]
        assert names == ["zebra", "alpha", "middle"]

        # Membership node variables should also be in definition order
        assert rel.expression.condition.variables == ["zebra", "alpha", "middle"]

    def test_many_columns_order(self):
        schema = "CREATE TABLE t (e INT, d INT, c INT, b INT, a INT);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["e", "d", "c", "b", "a"]


class TestTableNameVariants:
    """Test various table name formats."""

    def test_backtick_quoted_name(self):
        schema = "CREATE TABLE `my_table` (id INT, name TEXT);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].table_name == "my_table"

    def test_double_quoted_name(self):
        schema = 'CREATE TABLE "my_table" (id INT, name TEXT);'
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].table_name == "my_table"

    def test_case_insensitive_create_table(self):
        schema = "create table users (id INT, name TEXT);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].table_name == "users"

    def test_mixed_case_create_table(self):
        schema = "Create Table users (id INT, name TEXT);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].table_name == "users"

    def test_if_not_exists(self):
        schema = "CREATE TABLE IF NOT EXISTS users (id INT, name TEXT);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].table_name == "users"
        assert result.relations[0].columns == ["id", "name"]

    def test_no_trailing_semicolon(self):
        schema = "CREATE TABLE users (id INT, name TEXT)"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].table_name == "users"



class TestQuotedColumnNames:
    r"""Quoted column names preserve internal whitespace and punctuation.

    Background — run-15 dev_1298:
    BIRD's ``Examination`` schema has columns named with backticks:
    ``\`Examination Date\```, ``\`aCL IgG\```, ``\`aCL IgM\```,
    ``\`aCL IgA\```, ``\`ANA Pattern\``` — distinct columns whose
    bare-word prefix collides. The previous regex
    ``[`\"']?(\w+)[`\"']?`` only captured ``\w+`` (no spaces), so
    every multi-word backtick-quoted column collapsed to its first
    word, producing duplicate ``aCL`` and ``ANA`` entries in the
    column list. The duplicates corrupted the DRC's membership and
    indirectly drove the LLM's planner-loop into
    ``OPERATOR_SELECTION_FAILED``.
    """

    def test_backtick_quoted_column_with_space(self):
        schema = "CREATE TABLE t (`First Date` DATE, `Last Date` DATE);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        # Both names survive in full — no collision on the bare prefix.
        assert result.relations[0].columns == ["First Date", "Last Date"]

    def test_backtick_quoted_column_with_dash(self):
        r"""``\`T-CHO\``` keeps the dash — bare-word regex would have
        truncated it to ``T``."""
        schema = "CREATE TABLE t (id INT, `T-CHO` REAL, `TG` REAL);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["id", "T-CHO", "TG"]

    def test_double_quoted_column_with_space(self):
        schema = "CREATE TABLE t (\"Full Name\" TEXT, age INT);"
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        assert result.relations[0].columns == ["Full Name", "age"]

    def test_dev_1298_examination_schema_columns_are_distinct(self):
        """Reproduce the dev_1298 schema fragment — all six columns
        must show up as distinct entries (no duplicate ``aCL`` or
        ``ANA``)."""
        schema = (
            "CREATE TABLE Examination ("
            "  ID INTEGER, "
            "  `Examination Date` DATE, "
            "  `aCL IgG` REAL, "
            "  `aCL IgM` REAL, "
            "  ANA INTEGER, "
            "  `ANA Pattern` TEXT, "
            "  `aCL IgA` REAL"
            ");"
        )
        result = convert_tables(schema)

        assert isinstance(result, TableConversionSuccess)
        cols = result.relations[0].columns
        assert cols == [
            "ID",
            "Examination Date",
            "aCL IgG",
            "aCL IgM",
            "ANA",
            "ANA Pattern",
            "aCL IgA",
        ]
        # No duplicates — the run-15 bug produced three ``aCL`` entries
        # and two ``ANA`` entries.
        assert len(cols) == len(set(cols))

    def test_unterminated_backtick_returns_none(self):
        """Malformed input (open-backtick without close) is handled
        gracefully by skipping the column rather than raising."""
        # An unterminated backtick consumes the rest of the column
        # definition. The converter should treat the column as
        # un-extractable and either skip it or fall through. We
        # assert it doesn't crash — exact behaviour is implementation-
        # defined.
        schema = "CREATE TABLE t (`bad INT, name TEXT);"
        result = convert_tables(schema)
        # Either succeeds with whatever columns it could extract, or
        # fails cleanly. Not raising is the contract.
        assert isinstance(
            result, (TableConversionSuccess, TableConversionFailure)
        )
