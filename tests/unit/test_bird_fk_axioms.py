"""Unit tests for FK-aware equivalence checking in the BIRD runner.

BIRD often writes gold queries that ``INNER JOIN`` on foreign-key
columns purely to filter against a parent table — e.g. the gold for
``"What was the product_id of the transaction at 21:20?"`` joins
``transactions_1k`` with ``gasstations`` on ``GasStationID`` even
though the question doesn't involve gasstations at all. Without a
foreign-key axiom the equivalence checker correctly says these
queries differ (a row in ``transactions_1k`` *might* have a
``GasStationID`` that doesn't appear in ``gasstations``); with the
axiom it can prove them equivalent.

These tests cover the three pieces:

1. Loader: ``BirdLoader`` reads ``{split}_tables.json`` and resolves
   integer-index FKs into ``ForeignKey(from_table, from_column,
   to_table, to_column)``.
2. Axiom builder: ``_build_fk_axiom`` and ``_fk_axioms_from_test_case``
   produce SMT-LIB ``(assert (forall ...))`` strings of the right
   shape, with the right per-position sorts.
3. End-to-end: ``run_one`` threads the axioms through to the
   equivalence checker on both the original call and the
   projection-tolerance retry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird_benchmark.loader import BirdLoader, BirdLoaderConfig
from bird_benchmark.runner import (
    _build_fk_axiom,
    _fk_axioms_from_test_case,
    run_one,
)
from bird_benchmark.types import (
    ForeignKey,
    RunOptions,
    TestCase,
    Verdict,
)
from text_to_sql_planner.equivalence import (
    EquivalentResult,
    NotEquivalentResult,
)
from text_to_sql_planner.main import TextToSQLSuccess
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    MembershipNode,
)
from text_to_sql_planner.types.operation_tree import (
    OperationTree,
    TableLeafNode,
)


# ---------------------------------------------------------------------------
# Loader: BirdLoader reads dev_tables.json
# ---------------------------------------------------------------------------


def _write_minimal_install(
    bird_root: Path,
    *,
    tables_metadata: list[dict] | None,
):
    """Create a tiny BIRD-shaped install on disk with one record + one db.

    ``tables_metadata`` is the *contents* the loader will see at
    ``{split}_tables.json``; pass ``None`` to omit the file entirely.
    """

    import sqlite3

    split = "dev"
    split_dir = bird_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    db_dir = split_dir / f"{split}_databases" / "school"
    db_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = db_dir / "school.sqlite"
    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.execute("CREATE TABLE Departments (dept_id INT, name TEXT)")
        conn.execute("CREATE TABLE Employees (emp_id INT, name TEXT, dept_id INT)")
        conn.commit()

    record = {
        "question_id": 0,
        "db_id": "school",
        "question": "How many employees per department?",
        "evidence": "",
        "SQL": "SELECT COUNT(*) FROM Employees",
    }
    (split_dir / f"{split}.json").write_text(json.dumps([record]))

    if tables_metadata is not None:
        (split_dir / f"{split}_tables.json").write_text(
            json.dumps(tables_metadata)
        )


def test_loader_resolves_declared_foreign_keys(tmp_path):
    """The loader decodes BIRD's index-pair FK encoding into ``ForeignKey``."""

    # BIRD's convention:
    #   column_names_original[0] = (-1, "*")
    #   column_names_original[i] = (table_idx_into_table_names_original, name)
    # foreign_keys is a list of [from_col_idx, to_col_idx] pairs.
    metadata = [
        {
            "db_id": "school",
            "table_names_original": ["Departments", "Employees"],
            "column_names_original": [
                [-1, "*"],
                [0, "dept_id"],     # 1: Departments.dept_id
                [0, "name"],        # 2: Departments.name
                [1, "emp_id"],      # 3: Employees.emp_id
                [1, "name"],        # 4: Employees.name
                [1, "dept_id"],     # 5: Employees.dept_id
            ],
            "column_types": ["text", "integer", "text", "integer", "text", "integer"],
            "foreign_keys": [[5, 1]],  # Employees.dept_id -> Departments.dept_id
            "primary_keys": [1, 3],
        }
    ]
    _write_minimal_install(tmp_path, tables_metadata=metadata)

    loader = BirdLoader(BirdLoaderConfig(bird_root=tmp_path, split="dev"))
    cases = [c for c in loader.load() if isinstance(c, TestCase)]
    assert len(cases) == 1
    fks = cases[0].foreign_keys
    assert len(fks) == 1
    assert fks[0] == ForeignKey(
        from_table="Employees",
        from_column="dept_id",
        to_table="Departments",
        to_column="dept_id",
    )


def test_loader_yields_empty_fks_when_metadata_file_missing(tmp_path):
    """The metadata file is optional — its absence is non-fatal."""

    _write_minimal_install(tmp_path, tables_metadata=None)
    loader = BirdLoader(BirdLoaderConfig(bird_root=tmp_path, split="dev"))
    cases = [c for c in loader.load() if isinstance(c, TestCase)]
    assert len(cases) == 1
    assert cases[0].foreign_keys == []


def test_loader_skips_malformed_fk_indices(tmp_path):
    """Out-of-range / wrong-shape FK entries are silently skipped."""

    metadata = [
        {
            "db_id": "school",
            "table_names_original": ["Departments", "Employees"],
            "column_names_original": [
                [-1, "*"],
                [0, "dept_id"],
                [1, "dept_id"],
            ],
            "column_types": ["text", "integer", "integer"],
            "foreign_keys": [
                [2, 1],            # valid: Employees.dept_id -> Departments.dept_id
                [99, 1],           # bad: out of range
                ["not", "indices"], # bad: wrong types
                [0, 1],            # bad: index 0 is the synthetic "*" row
            ],
            "primary_keys": [1],
        }
    ]
    _write_minimal_install(tmp_path, tables_metadata=metadata)
    loader = BirdLoader(BirdLoaderConfig(bird_root=tmp_path, split="dev"))
    cases = [c for c in loader.load() if isinstance(c, TestCase)]
    assert len(cases[0].foreign_keys) == 1


def test_loader_handles_malformed_metadata_json(tmp_path):
    """Garbled JSON in the metadata file degrades to empty FK lists."""

    _write_minimal_install(tmp_path, tables_metadata=[])
    # Overwrite the metadata file with garbage.
    (tmp_path / "dev" / "dev_tables.json").write_text("not valid json{{{")
    loader = BirdLoader(BirdLoaderConfig(bird_root=tmp_path, split="dev"))
    cases = [c for c in loader.load() if isinstance(c, TestCase)]
    assert cases[0].foreign_keys == []


# ---------------------------------------------------------------------------
# Axiom builder: shape and per-position sorts
# ---------------------------------------------------------------------------


def test_build_fk_axiom_shape_with_string_sorts():
    """A single FK produces a forall/=>/exists axiom with the right sorts."""

    # Employees(emp_id INT, name TEXT, dept_id INT) -- FK on dept_id.
    # Departments(dept_id INT, name TEXT)
    axiom = _build_fk_axiom(
        from_table="Employees",
        from_columns=["emp_id", "name", "dept_id"],
        from_types={"emp_id": "Int", "name": "String", "dept_id": "Int"},
        from_idx=2,
        to_table="Departments",
        to_columns=["dept_id", "name"],
        to_types={"dept_id": "Int", "name": "String"},
        to_idx=0,
    )

    # Universal: one var per Employees column, each with the right sort.
    assert "(_fk_Employees_emp_id Int)" in axiom
    assert "(_fk_Employees_name String)" in axiom
    assert "(_fk_Employees_dept_id Int)" in axiom

    # Antecedent: Employees(_fk_Employees_emp_id _fk_Employees_name _fk_Employees_dept_id)
    assert (
        "(Employees _fk_Employees_emp_id _fk_Employees_name _fk_Employees_dept_id)"
        in axiom
    )

    # Existential: every Departments column EXCEPT dept_id (the FK
    # position), which is replaced by the FK var from the universal.
    assert "(_fk_Departments_name String)" in axiom
    assert "(_fk_Departments_dept_id" not in axiom  # NOT existentially bound
    # The Departments call uses the universal var at position 0 (FK):
    assert (
        "(Departments _fk_Employees_dept_id _fk_Departments_name)"
        in axiom
    )

    # Implication structure.
    assert axiom.startswith("(assert (forall (")
    assert " (=> " in axiom
    assert axiom.endswith(")")


def test_build_fk_axiom_handles_singleton_target_table():
    """A target table with only the FK column has no existential body."""

    axiom = _build_fk_axiom(
        from_table="Child",
        from_columns=["fk_id"],
        from_types={"fk_id": "Int"},
        from_idx=0,
        to_table="Parent",
        to_columns=["pk_id"],
        to_types={"pk_id": "Int"},
        to_idx=0,
    )
    # No existential needed when every target column IS the FK column.
    assert "exists" not in axiom
    assert "(Parent _fk_Child_fk_id)" in axiom


def test_build_fk_axiom_defaults_unknown_columns_to_int():
    """Columns not in the type map fall back to Int — the SMT default."""

    axiom = _build_fk_axiom(
        from_table="A",
        from_columns=["x", "y"],
        from_types={},  # no entries — fall back to Int
        from_idx=0,
        to_table="B",
        to_columns=["k", "v"],
        to_types={},
        to_idx=0,
    )
    assert "(_fk_A_x Int)" in axiom
    assert "(_fk_A_y Int)" in axiom
    assert "(_fk_B_v Int)" in axiom


def test_fk_axioms_from_test_case_renders_one_axiom_per_fk():
    """``_fk_axioms_from_test_case`` produces one axiom per declared FK."""

    case = TestCase(
        test_case_id="t1",
        split="dev",
        db_id="school",
        schema=(
            "CREATE TABLE Departments (dept_id INT, name TEXT);"
            "CREATE TABLE Employees (emp_id INT, name TEXT, dept_id INT);"
        ),
        question="?",
        evidence="",
        gold_sql="SELECT 1",
        foreign_keys=[
            ForeignKey(
                from_table="Employees",
                from_column="dept_id",
                to_table="Departments",
                to_column="dept_id",
            )
        ],
    )
    axioms = _fk_axioms_from_test_case(case)
    assert len(axioms) == 1
    assert "Employees" in axioms[0]
    assert "Departments" in axioms[0]


def test_fk_axioms_from_test_case_skips_unknown_table():
    """An FK referencing a table not in the schema is silently skipped."""

    case = TestCase(
        test_case_id="t1",
        split="dev",
        db_id="school",
        schema="CREATE TABLE Employees (emp_id INT, name TEXT)",
        question="?",
        evidence="",
        gold_sql="SELECT 1",
        foreign_keys=[
            ForeignKey(
                from_table="Employees",
                from_column="dept_id",
                to_table="Departments",  # not in schema
                to_column="dept_id",
            )
        ],
    )
    assert _fk_axioms_from_test_case(case) == []


def test_fk_axioms_returns_empty_when_no_fks():
    """A test case with no FKs produces no axioms."""

    case = TestCase(
        test_case_id="t1",
        split="dev",
        db_id="school",
        schema="CREATE TABLE X (a INT)",
        question="?",
        evidence="",
        gold_sql="SELECT 1",
        foreign_keys=[],
    )
    assert _fk_axioms_from_test_case(case) == []


def test_fk_axioms_match_table_name_case_insensitively():
    """BIRD's metadata can disagree on case with the SQL DDL — match anyway."""

    case = TestCase(
        test_case_id="t1",
        split="dev",
        db_id="school",
        schema=(
            'CREATE TABLE "EMPLOYEES" (emp_id INT, dept_id INT);'
            'CREATE TABLE "DEPARTMENTS" (dept_id INT);'
        ),
        question="?",
        evidence="",
        gold_sql="SELECT 1",
        foreign_keys=[
            ForeignKey(
                from_table="employees",        # lowercase
                from_column="dept_id",
                to_table="departments",
                to_column="dept_id",
            )
        ],
    )
    axioms = _fk_axioms_from_test_case(case)
    assert len(axioms) == 1


# ---------------------------------------------------------------------------
# End-to-end: run_one threads axioms through to the equivalence checker
# ---------------------------------------------------------------------------


def _make_case_with_fks() -> TestCase:
    return TestCase(
        test_case_id="dev_99",
        split="dev",
        db_id="school",
        schema=(
            "CREATE TABLE Departments (dept_id INT, name TEXT);"
            "CREATE TABLE Employees (emp_id INT, name TEXT, dept_id INT);"
        ),
        question="How many employees?",
        evidence="",
        gold_sql="SELECT COUNT(*) FROM Employees",
        foreign_keys=[
            ForeignKey(
                from_table="Employees",
                from_column="dept_id",
                to_table="Departments",
                to_column="dept_id",
            )
        ],
    )


def _make_options() -> RunOptions:
    return RunOptions(
        bird_root=Path("/tmp/bird"),
        split="dev",
        cvc5_timeout_seconds=30,
        per_test_timeout_seconds=60,
    )


def _make_drc() -> DRCExpression:
    return DRCExpression(
        result_variables=[ColumnVariable(name="emp_id")],
        condition=MembershipNode(
            variables=["emp_id", "name", "dept_id"],
            relation="Employees",
        ),
    )


def _make_planner_success() -> TextToSQLSuccess:
    drc = _make_drc()
    tree = OperationTree(root=TableLeafNode(table_name="Employees"))
    return TextToSQLSuccess(
        sql="SELECT emp_id FROM Employees",
        operation_tree=tree,
        target_expression=drc,
        target_query=drc,
    )


def _planner_returning(result):
    async def _run(**kwargs):
        return result

    return _run


def _converter_returning(result):
    def _convert(sql, schema):
        return result

    return _convert


@pytest.mark.asyncio
async def test_run_one_passes_fk_axioms_to_equivalence_check():
    """The runner derives FK axioms from the test case and threads them through."""

    captured: list[list[str] | None] = []

    async def _capturing_check(expr1, expr2, config, *, schema_types=None, axioms=None, **kwargs):
        captured.append(axioms)
        return EquivalentResult()

    rr = await run_one(
        _make_case_with_fks(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_capturing_check,
    )

    assert rr.underlying_verdict == Verdict.equivalent
    assert len(captured) == 1
    axioms = captured[0]
    assert axioms is not None
    assert len(axioms) == 1
    # Sanity-check the axiom mentions both tables.
    assert "Employees" in axioms[0]
    assert "Departments" in axioms[0]


@pytest.mark.asyncio
async def test_run_one_passes_fk_axioms_to_projection_retry():
    """Both the original equivalence call and the prefix-tolerance retry get the same axioms."""

    captured_kwargs: list[dict] = []

    async def _switching_check(expr1, expr2, config, *, schema_types=None, axioms=None, **kwargs):
        captured_kwargs.append({"axioms": axioms})
        # First call -> not_equivalent, second call -> equivalent. The
        # truncation retry path must receive the same axioms as the
        # first call so referential-integrity facts survive the retry.
        if len(captured_kwargs) == 1:
            return NotEquivalentResult()
        return EquivalentResult()

    # Generated has 2 result vars, gold has 1 — the prefix retry fires.
    from text_to_sql_planner.types.drc import AggregateVariable

    generated = DRCExpression(
        result_variables=[
            ColumnVariable(name="emp_id"),
            AggregateVariable(function="SUM", column="dept_id"),
        ],
        condition=MembershipNode(
            variables=["emp_id", "name", "dept_id"],
            relation="Employees",
        ),
    )
    gold = DRCExpression(
        result_variables=[ColumnVariable(name="emp_id")],
        condition=MembershipNode(
            variables=["emp_id", "name", "dept_id"],
            relation="Employees",
        ),
    )

    success = TextToSQLSuccess(
        sql="SELECT emp_id, SUM(dept_id) FROM Employees GROUP BY emp_id",
        operation_tree=OperationTree(root=TableLeafNode(table_name="Employees")),
        target_expression=generated,
        target_query=generated,
    )

    rr = await run_one(
        _make_case_with_fks(),
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(success),
        convert_sql_callable=_converter_returning(gold),
        check_equivalence_callable=_switching_check,
    )

    assert rr.underlying_verdict == Verdict.equivalent
    assert len(captured_kwargs) == 2
    # Axioms must be identical across the two calls.
    assert captured_kwargs[0]["axioms"] == captured_kwargs[1]["axioms"]
    assert captured_kwargs[0]["axioms"] is not None
    assert len(captured_kwargs[0]["axioms"]) == 1


@pytest.mark.asyncio
async def test_run_one_passes_empty_axioms_when_no_fks():
    """Test cases without FK metadata still pass an empty list, not None."""

    case = TestCase(
        test_case_id="dev_99",
        split="dev",
        db_id="school",
        schema="CREATE TABLE X (a INT)",
        question="?",
        evidence="",
        gold_sql="SELECT 1",
        foreign_keys=[],
    )

    captured: list[list[str] | None] = []

    async def _capturing_check(expr1, expr2, config, *, schema_types=None, axioms=None, **kwargs):
        captured.append(axioms)
        return EquivalentResult()

    await run_one(
        case,
        _make_options(),
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_capturing_check,
    )
    # Empty list, not None — the axioms parameter is always passed
    # by the runner so callers can rely on it.
    assert captured == [[]]
