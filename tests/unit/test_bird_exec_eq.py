"""Unit tests for the BIRD execution-equivalence check.

The runner now executes both the generated SQL and the gold SQL
against the per-Test_Case SQLite database and reports whether the
result sets agree, in addition to the cvc5 logical equivalence
verdict. These are complementary signals: the cvc5 verdict is a
proof over all DB states; the execution verdict is an observation
on the one DB state BIRD ships.

These tests cover the three pieces:

1. ``compare_executions`` against a real on-disk SQLite DB.
2. ``_finalize_with_execution`` skip logic — ``execution_check``
   off, empty SQL, and missing DB don't raise.
3. End-to-end through ``run_one`` that the
   :class:`~bird_benchmark.types.ExecutionResult` lands on
   :attr:`~bird_benchmark.types.RunResult.execution`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bird_benchmark.exec_eq import compare_executions
from bird_benchmark.runner import (
    _finalize_with_execution,
    _sqlite_path_for,
    run_one,
)
from bird_benchmark.types import (
    ExecutionResult,
    ExecutionStatus,
    RunOptions,
    RunResult,
    TestCase,
    Verdict,
)
from text_to_sql_planner.equivalence import EquivalentResult, NotEquivalentResult
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
# compare_executions: real SQLite, real query execution
# ---------------------------------------------------------------------------


def _make_minimal_db(tmp_path: Path) -> Path:
    """Create a 3-row SQLite database used by all the per-query tests."""

    db = tmp_path / "school.sqlite"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("CREATE TABLE Employees (id INT, name TEXT, dept INT)")
        conn.executemany(
            "INSERT INTO Employees VALUES (?, ?, ?)",
            [(1, "Alice", 10), (2, "Bob", 20), (3, "Carol", 10)],
        )
        conn.commit()
    return db


def test_compare_executions_match(tmp_path):
    """Two equivalent queries return ``status=match``, both flags True."""
    db = _make_minimal_db(tmp_path)
    result = compare_executions(
        sqlite_path=db,
        generated_sql="SELECT id FROM Employees WHERE dept = 10",
        gold_sql="SELECT id FROM Employees WHERE dept = 10 ORDER BY id",
        timeout_seconds=5.0,
    )
    assert result.status == ExecutionStatus.match
    assert result.set_match is True
    assert result.multiset_match is True
    assert result.generated_row_count == 2
    assert result.gold_row_count == 2


def test_compare_executions_mismatch(tmp_path):
    """Two queries returning different rows → ``status=mismatch``."""
    db = _make_minimal_db(tmp_path)
    result = compare_executions(
        sqlite_path=db,
        generated_sql="SELECT id FROM Employees WHERE dept = 10",
        gold_sql="SELECT id FROM Employees WHERE dept = 20",
        timeout_seconds=5.0,
    )
    assert result.status == ExecutionStatus.mismatch
    assert result.set_match is False
    assert result.multiset_match is False


def test_compare_executions_set_match_but_multiset_mismatch(tmp_path):
    """When duplicates differ, ``set_match`` wins but ``multiset_match`` reveals the bug."""
    db = _make_minimal_db(tmp_path)
    # Generated: returns each dept once via DISTINCT.
    # Gold: returns each dept once per row (duplicates).
    result = compare_executions(
        sqlite_path=db,
        generated_sql="SELECT DISTINCT dept FROM Employees",
        gold_sql="SELECT dept FROM Employees",
        timeout_seconds=5.0,
    )
    # Set equality: {10, 20} == {10, 20} → match. Multiset: differs.
    assert result.status == ExecutionStatus.match
    assert result.set_match is True
    assert result.multiset_match is False
    assert result.generated_row_count == 2
    assert result.gold_row_count == 3


def test_compare_executions_generated_error(tmp_path):
    """A SQL error on the generated side surfaces as ``generated_error``."""
    db = _make_minimal_db(tmp_path)
    result = compare_executions(
        sqlite_path=db,
        generated_sql="SELECT not_a_column FROM Employees",
        gold_sql="SELECT id FROM Employees",
        timeout_seconds=5.0,
    )
    assert result.status == ExecutionStatus.generated_error
    assert "not_a_column" in result.error or "no such column" in result.error.lower()


def test_compare_executions_gold_error(tmp_path):
    """A SQL error on the gold side surfaces as ``gold_error``."""
    db = _make_minimal_db(tmp_path)
    result = compare_executions(
        sqlite_path=db,
        generated_sql="SELECT id FROM Employees",
        gold_sql="SELECT * FROM nonexistent_table",
        timeout_seconds=5.0,
    )
    assert result.status == ExecutionStatus.gold_error
    assert "nonexistent_table" in result.error.lower() or "no such table" in result.error.lower()


def test_compare_executions_db_unavailable(tmp_path):
    """A missing SQLite file surfaces as ``db_unavailable``."""
    result = compare_executions(
        sqlite_path=tmp_path / "missing.sqlite",
        generated_sql="SELECT 1",
        gold_sql="SELECT 1",
        timeout_seconds=5.0,
    )
    assert result.status == ExecutionStatus.db_unavailable
    assert "missing.sqlite" in result.error


def test_compare_executions_empty_sql(tmp_path):
    """Empty SQL on either side skips the check entirely."""
    db = _make_minimal_db(tmp_path)
    result = compare_executions(
        sqlite_path=db,
        generated_sql="",
        gold_sql="SELECT 1",
        timeout_seconds=5.0,
    )
    assert result.status == ExecutionStatus.skipped


def test_compare_executions_read_only_mode_rejects_writes(tmp_path):
    """Read-only URI mode prevents a malicious query from damaging the DB."""
    db = _make_minimal_db(tmp_path)
    # An UPDATE on a read-only connection raises sqlite3.OperationalError.
    result = compare_executions(
        sqlite_path=db,
        generated_sql="UPDATE Employees SET name = 'pwn3d' WHERE id = 1",
        gold_sql="SELECT id FROM Employees",
        timeout_seconds=5.0,
    )
    # The update fails on the generated side; the framework reports it
    # as a generated_error rather than silently ignoring it.
    assert result.status == ExecutionStatus.generated_error
    # Confirm the data is unchanged.
    with sqlite3.connect(str(db)) as conn:
        cursor = conn.execute("SELECT name FROM Employees WHERE id = 1")
        assert cursor.fetchone() == ("Alice",)


# ---------------------------------------------------------------------------
# _sqlite_path_for: path derivation matches the loader
# ---------------------------------------------------------------------------


def test_sqlite_path_for_matches_loader_convention(tmp_path):
    """The path the runner builds matches the loader's ``_sqlite_path``."""
    case = TestCase(
        test_case_id="dev_42",
        split="dev",
        db_id="financial",
        schema="",
        question="?",
        evidence="",
        gold_sql="",
    )
    options = RunOptions(bird_root=tmp_path, split="dev")
    expected = (
        tmp_path / "dev" / "dev_databases" / "financial" / "financial.sqlite"
    )
    assert _sqlite_path_for(case, options) == expected


# ---------------------------------------------------------------------------
# _finalize_with_execution: skip behaviour
# ---------------------------------------------------------------------------


def _make_result(*, generated_sql: str = "SELECT 1", gold_sql: str = "SELECT 1") -> RunResult:
    return RunResult(
        test_case_id="dev_1",
        underlying_verdict=Verdict.equivalent,
        reported_verdict=Verdict.equivalent,
        question="?",
        evidence="",
        planner_input="?",
        generated_sql=generated_sql,
        gold_sql=gold_sql,
    )


def _make_test_case() -> TestCase:
    return TestCase(
        test_case_id="dev_1",
        split="dev",
        db_id="school",
        schema="",
        question="?",
        evidence="",
        gold_sql="SELECT 1",
    )


def test_finalize_skips_when_execution_check_disabled():
    """``execution_check=False`` leaves ``execution=None`` and never invokes the executor."""
    options = RunOptions(
        bird_root=Path("/tmp"), split="dev", execution_check=False
    )
    invoked = []

    def fake_exec(**kwargs):
        invoked.append(kwargs)
        raise AssertionError("executor must not be called when check is disabled")

    result = _finalize_with_execution(
        _make_test_case(), options, _make_result(),
        exec_callable=fake_exec,
    )
    assert result.execution is None
    assert invoked == []


def test_finalize_skips_when_generated_sql_empty():
    """Empty generated SQL → status=skipped, executor not invoked."""
    options = RunOptions(bird_root=Path("/tmp"), split="dev")
    invoked = []

    def fake_exec(**kwargs):
        invoked.append(kwargs)
        raise AssertionError("executor must not be called for empty SQL")

    result = _finalize_with_execution(
        _make_test_case(), options,
        _make_result(generated_sql=""),
        exec_callable=fake_exec,
    )
    assert result.execution is not None
    assert result.execution.status == ExecutionStatus.skipped
    assert invoked == []


def test_finalize_invokes_executor_when_sql_present():
    """The executor is called with the right path/SQL/timeout."""
    options = RunOptions(
        bird_root=Path("/tmp/bird"),
        split="dev",
        execution_timeout_seconds=15,
    )
    captured: dict = {}

    def fake_exec(**kwargs):
        captured.update(kwargs)
        return ExecutionResult(
            status=ExecutionStatus.match, set_match=True, multiset_match=True,
        )

    result = _finalize_with_execution(
        _make_test_case(), options,
        _make_result(generated_sql="SELECT id FROM t", gold_sql="SELECT id FROM t"),
        exec_callable=fake_exec,
    )
    assert result.execution is not None
    assert result.execution.status == ExecutionStatus.match
    # The runner derived the path from options + test_case.db_id.
    assert captured["sqlite_path"] == (
        Path("/tmp/bird/dev/dev_databases/school/school.sqlite")
    )
    assert captured["generated_sql"] == "SELECT id FROM t"
    assert captured["gold_sql"] == "SELECT id FROM t"
    assert captured["timeout_seconds"] == 15.0


def test_finalize_swallows_executor_exceptions():
    """A bug in the executor must not crash run_one."""
    options = RunOptions(bird_root=Path("/tmp"), split="dev")

    def crashing_exec(**kwargs):
        raise RuntimeError("boom")

    result = _finalize_with_execution(
        _make_test_case(), options, _make_result(),
        exec_callable=crashing_exec,
    )
    assert result.execution is not None
    assert result.execution.status == ExecutionStatus.db_unavailable
    assert "RuntimeError: boom" in result.execution.error


# ---------------------------------------------------------------------------
# End-to-end: run_one populates RunResult.execution
# ---------------------------------------------------------------------------


def _make_drc() -> DRCExpression:
    return DRCExpression(
        result_variables=[ColumnVariable(name="id")],
        condition=MembershipNode(variables=["id"], relation="t"),
    )


def _planner_returning(result):
    async def _run(**kwargs):
        return result

    return _run


def _converter_returning(result):
    def _convert(sql, schema):
        return result

    return _convert


def _equivalence_returning(result):
    async def _check(*args, **kwargs):
        return result

    return _check


def _make_planner_success() -> TextToSQLSuccess:
    drc = _make_drc()
    tree = OperationTree(root=TableLeafNode(table_name="t"))
    return TextToSQLSuccess(
        sql="SELECT id FROM t",
        operation_tree=tree,
        target_expression=drc,
        target_query=drc,
    )


@pytest.mark.asyncio
async def test_run_one_attaches_execution_result_to_run_result():
    """A successful run_one populates ``RunResult.execution`` from the executor."""
    options = RunOptions(bird_root=Path("/tmp"), split="dev")
    case = _make_test_case()

    def fake_exec(**kwargs):
        return ExecutionResult(
            status=ExecutionStatus.match,
            set_match=True,
            multiset_match=True,
            generated_row_count=5,
            gold_row_count=5,
        )

    rr = await run_one(
        case, options,
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=fake_exec,
    )

    assert rr.underlying_verdict == Verdict.equivalent
    assert rr.execution is not None
    assert rr.execution.status == ExecutionStatus.match
    assert rr.execution.generated_row_count == 5


@pytest.mark.asyncio
async def test_run_one_records_disagreement_between_cvc5_and_execution():
    """The dev_1519 pattern: cvc5 says not_equivalent, exec says match.

    This is the most useful diagnostic the dual-check produces — when
    cvc5 says the queries differ but they return the same rows on
    BIRD's snapshot, the gold is over-specified.
    """
    options = RunOptions(bird_root=Path("/tmp"), split="dev")
    case = _make_test_case()

    def fake_exec(**kwargs):
        return ExecutionResult(
            status=ExecutionStatus.match,
            set_match=True,
            multiset_match=True,
            generated_row_count=1,
            gold_row_count=1,
        )

    rr = await run_one(
        case, options,
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(NotEquivalentResult()),
        exec_callable=fake_exec,
    )

    # Both signals are recorded independently.
    assert rr.underlying_verdict == Verdict.not_equivalent
    assert rr.execution is not None
    assert rr.execution.status == ExecutionStatus.match


@pytest.mark.asyncio
async def test_run_one_records_skipped_execution_for_planner_failure():
    """When the planner fails, exec is recorded as ``skipped`` (no SQL to run)."""
    from text_to_sql_planner.main import TextToSQLFailure
    from text_to_sql_planner.types.errors import ErrorCode

    options = RunOptions(bird_root=Path("/tmp"), split="dev")

    rr = await run_one(
        _make_test_case(), options,
        expected_fail=set(),
        planner_callable=_planner_returning(
            TextToSQLFailure(
                error="planner gave up",
                code=ErrorCode.MAX_ITERATIONS_EXCEEDED,
            )
        ),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        # No fake_exec — defaults to compare_executions, but we
        # never call it because generated_sql is empty.
    )
    assert rr.underlying_verdict == Verdict.planner_failed
    assert rr.execution is not None
    assert rr.execution.status == ExecutionStatus.skipped


@pytest.mark.asyncio
async def test_run_one_skips_execution_when_check_disabled():
    """``options.execution_check=False`` leaves ``execution=None``."""
    options = RunOptions(
        bird_root=Path("/tmp"), split="dev", execution_check=False
    )

    def fake_exec(**kwargs):
        raise AssertionError("must not run with execution_check=False")

    rr = await run_one(
        _make_test_case(), options,
        expected_fail=set(),
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=fake_exec,
    )
    assert rr.execution is None


# ---------------------------------------------------------------------------
# CLI integration: --no-execution-check / --exec-timeout flow into options
# ---------------------------------------------------------------------------


def test_cli_default_enables_execution_check():
    """The default invocation has ``execution_check=True``."""
    from bird_benchmark.cli import _build_parser, _options_from_args

    parser = _build_parser()
    args = parser.parse_args(
        ["single", "--split", "dev", "--bird-root", "/tmp/bird", "--id", "dev_1"]
    )
    options = _options_from_args(args)
    assert options.execution_check is True
    assert options.execution_timeout_seconds == 30


def test_cli_no_execution_check_disables_it():
    """``--no-execution-check`` flips the flag to False."""
    from bird_benchmark.cli import _build_parser, _options_from_args

    parser = _build_parser()
    args = parser.parse_args(
        [
            "single",
            "--split", "dev", "--bird-root", "/tmp/bird", "--id", "dev_1",
            "--no-execution-check",
        ]
    )
    options = _options_from_args(args)
    assert options.execution_check is False


def test_cli_exec_timeout_override():
    """``--exec-timeout`` flows into ``execution_timeout_seconds``."""
    from bird_benchmark.cli import _build_parser, _options_from_args

    parser = _build_parser()
    args = parser.parse_args(
        [
            "single",
            "--split", "dev", "--bird-root", "/tmp/bird", "--id", "dev_1",
            "--exec-timeout", "120",
        ]
    )
    options = _options_from_args(args)
    assert options.execution_timeout_seconds == 120
