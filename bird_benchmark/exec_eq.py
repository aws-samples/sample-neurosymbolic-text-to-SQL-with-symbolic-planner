"""Execution-equivalence check for BIRD test cases.

In addition to logical equivalence (cvc5 over DRC), the runner can
execute both the generated SQL and the gold SQL against BIRD's
per-database SQLite snapshot and compare result sets. This is what
BIRD's official leaderboard does.

The two signals are complementary, not redundant:

* **Logical equivalence** is a proof over *all* possible database
  states. cvc5 says ``unsat`` only when the queries cannot disagree
  on any input; that's a strong guarantee but it's also strict —
  declared-but-not-enforced foreign-key joins, redundant filters,
  and other "vacuous on this database" patterns will be flagged as
  not-equivalent even when the queries return the same rows here.

* **Execution equivalence** is an observation on the one database
  BIRD ships. It catches the patterns that logical equivalence is
  too strict for, and it directly mirrors BIRD's own evaluation
  protocol so framework numbers can be compared against published
  results. But it's only as strong as the database used: a query
  that depends on a corner case absent from BIRD's snapshot can
  pass execution and still be wrong.

Reporting both signals lets an operator see the *disagreement*,
which is itself the most useful diagnostic — a "logical:
not_equivalent, execution: match" verdict is the fingerprint of an
over-specified BIRD gold query.

API
---

The public entry point is :func:`compare_executions`. It opens the
SQLite database read-only, runs both queries with a per-query
watchdog, and returns an :class:`~bird_benchmark.types.ExecutionResult`.
It never raises — every error path is encoded in the
:class:`~bird_benchmark.types.ExecutionStatus` and the ``error``
field, so the runner can keep going on a malformed query.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from bird_benchmark.types import ExecutionResult, ExecutionStatus


# Result-set comparison happens on tuples of native Python objects
# (the values sqlite3 returns by default). We don't normalise types
# beyond that — if the gold query returns ``1`` (int) and the
# generated query returns ``"1"`` (str), they should not compare
# equal, because the generated query is doing something different.
# This matches BIRD's own evaluator.


# Cap row counts so a runaway query that returns the entire DB
# doesn't blow up memory in the runner. The cap is generous; a
# Test_Case whose answer is genuinely > 100k rows is broken anyway.
_MAX_ROWS = 100_000


def compare_executions(
    *,
    sqlite_path: Path,
    generated_sql: str,
    gold_sql: str,
    timeout_seconds: float,
) -> ExecutionResult:
    """Run both queries on the BIRD database and compare result sets.

    Parameters
    ----------
    sqlite_path:
        Path to the per-database SQLite file (the loader knows how
        to derive this; the runner builds it from
        ``options.bird_root`` and ``test_case.db_id``).
    generated_sql:
        The planner's SQL output. Empty string skips the check —
        the runner sets the status to ``skipped``.
    gold_sql:
        BIRD's gold SQL.
    timeout_seconds:
        Per-query wall-clock budget. Enforced via
        :meth:`sqlite3.Connection.interrupt` from a daemon
        watchdog thread; the underlying SQLite call returns
        :class:`sqlite3.OperationalError("interrupted")` when the
        watchdog fires.

    Returns
    -------
    ExecutionResult
        Always populated. ``status`` summarises the outcome;
        ``multiset_match`` and ``set_match`` are meaningful only
        when ``status == match`` or ``status == mismatch``.
    """

    if not generated_sql or not gold_sql:
        return ExecutionResult(
            status=ExecutionStatus.skipped,
            error="empty SQL on one or both sides; nothing to execute",
        )

    if not sqlite_path.is_file():
        return ExecutionResult(
            status=ExecutionStatus.db_unavailable,
            error=f"sqlite database not found at {sqlite_path}",
        )

    # Read-only URI mode so a malformed query can't damage the DB,
    # and so concurrent runs don't fight for write locks. ``immutable``
    # would be even faster but it locks out the connection from
    # ``interrupt()``, which we need for the watchdog.
    connection: sqlite3.Connection
    try:
        connection = sqlite3.connect(
            f"file:{sqlite_path}?mode=ro",
            uri=True,
            timeout=timeout_seconds,
        )
    except sqlite3.Error as exc:
        return ExecutionResult(
            status=ExecutionStatus.db_unavailable,
            error=f"failed to open {sqlite_path}: {exc}",
        )

    try:
        gen_outcome = _run_one(connection, generated_sql, timeout_seconds)
        gold_outcome = _run_one(connection, gold_sql, timeout_seconds)
    finally:
        connection.close()

    # Per-side error / timeout pre-empts comparison.
    if isinstance(gen_outcome, _Error):
        if gen_outcome.timed_out:
            return ExecutionResult(
                status=ExecutionStatus.timeout,
                error=f"generated query timed out: {gen_outcome.message}",
            )
        return ExecutionResult(
            status=ExecutionStatus.generated_error,
            error=gen_outcome.message,
        )
    if isinstance(gold_outcome, _Error):
        if gold_outcome.timed_out:
            return ExecutionResult(
                status=ExecutionStatus.timeout,
                error=f"gold query timed out: {gold_outcome.message}",
                generated_row_count=len(gen_outcome.rows),
            )
        return ExecutionResult(
            status=ExecutionStatus.gold_error,
            error=gold_outcome.message,
            generated_row_count=len(gen_outcome.rows),
        )

    return _compare_rows(gen_outcome.rows, gold_outcome.rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Rows:
    __slots__ = ("rows",)

    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows


class _Error:
    __slots__ = ("message", "timed_out")

    def __init__(self, message: str, *, timed_out: bool = False) -> None:
        self.message = message
        self.timed_out = timed_out


def _run_one(
    connection: sqlite3.Connection, sql: str, timeout_seconds: float
) -> _Rows | _Error:
    """Run a single query under a watchdog and return rows or a structured error.

    The watchdog is a daemon ``Timer`` that calls
    :meth:`sqlite3.Connection.interrupt` after ``timeout_seconds``.
    SQLite then aborts the in-flight query and the cursor's
    ``fetchall`` raises :class:`sqlite3.OperationalError("interrupted")``,
    which we map to a timeout error.

    Memory: we cap the result at ``_MAX_ROWS`` to bound memory in
    the runner. Genuinely-huge result sets are pathological for a
    BIRD Test_Case; treating the cap as a generated_error keeps the
    invariant that the runner uses bounded memory per Test_Case.
    """

    timed_out = threading.Event()

    def _interrupt() -> None:
        timed_out.set()
        try:
            connection.interrupt()
        except sqlite3.Error:
            # Connection might have closed between the timer firing
            # and the call. Suppress — the watchdog's job is best-effort.
            pass

    timer = threading.Timer(timeout_seconds, _interrupt)
    timer.daemon = True
    timer.start()
    try:
        cursor = connection.execute(sql)
    except sqlite3.Error as exc:
        timer.cancel()
        if timed_out.is_set():
            return _Error(str(exc), timed_out=True)
        return _Error(f"{type(exc).__name__}: {exc}")

    try:
        rows: list[tuple] = []
        for row in cursor:
            rows.append(tuple(row))
            if len(rows) > _MAX_ROWS:
                return _Error(
                    f"query returned more than {_MAX_ROWS} rows; truncating "
                    "and treating as error to bound runner memory",
                )
        return _Rows(rows)
    except sqlite3.Error as exc:
        if timed_out.is_set():
            return _Error(str(exc), timed_out=True)
        return _Error(f"{type(exc).__name__}: {exc}")
    finally:
        timer.cancel()
        try:
            cursor.close()
        except sqlite3.Error:
            pass


def _compare_rows(
    generated_rows: list[tuple], gold_rows: list[tuple]
) -> ExecutionResult:
    """Compare two result sets as multisets and as sets.

    Multiset comparison uses :class:`collections.Counter` keyed on
    the tuple itself. Set comparison collapses to ``set(...)``. We
    report both flags so the operator can tell apart "wrong rows"
    from "right rows but wrong duplicate count" — BIRD's official
    evaluator only checks set equality, but the multiset signal is
    strictly stricter and catches an extra class of bug for free.
    """

    from collections import Counter

    gen_count = Counter(generated_rows)
    gold_count = Counter(gold_rows)
    multiset_match = gen_count == gold_count
    set_match = set(generated_rows) == set(gold_rows)

    status = ExecutionStatus.match if set_match else ExecutionStatus.mismatch
    return ExecutionResult(
        status=status,
        multiset_match=multiset_match,
        set_match=set_match,
        generated_row_count=len(generated_rows),
        gold_row_count=len(gold_rows),
    )


__all__ = ["compare_executions"]
