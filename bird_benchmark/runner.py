"""Per-Test_Case orchestration for the BIRD benchmark framework.

This module implements :func:`run_one`, the load-bearing inner loop the
Single_Run and Suite_Run drivers (tasks 10.1 / 12.1) wrap. ``run_one``
takes one ``TestCase`` plus a ``RunOptions`` and returns a single
``RunResult``; the manifest, the progress lines, and the report are
the suite driver's job.

The per-Test_Case sequence (from ``design.md``):

1. Build the planner input via :func:`bird_benchmark.evidence.forward_call`,
   which routes the BIRD evidence string through a dedicated keyword
   parameter on the planner when one exists, or concatenates with a
   single ``\\n`` otherwise.
2. ``await asyncio.wait_for(planner_call,
   timeout=options.per_test_timeout_seconds)``. ``TimeoutError`` becomes
   ``Verdict.planner_failed`` with reason ``"planner_timeout"`` (Req
   6.6); a ``TextToSQLFailure`` becomes ``Verdict.planner_failed`` with
   the planner's ``error`` + ``code`` (Req 3.3 / 5.7).
3. Convert the gold SQL via
   :func:`bird_benchmark.sql_to_drc.convert_sql`. A
   ``ConverterError(kind="unsupported_feature")`` becomes
   ``Verdict.gold_conversion_failure`` (Req 5.8); a ``parse_error``
   becomes ``Verdict.skipped`` (Req 3.2).
4. Build an :class:`~text_to_sql_planner.equivalence.EquivalenceCheckerConfig`
   from the run options and call ``check_equivalence``. Map
   :class:`EquivalenceResult` to :class:`~bird_benchmark.types.Verdict`:

   ============================================  =================
   ``EquivalenceResult.status`` (and reason)     Verdict
   ============================================  =================
   ``"equivalent"``                              ``equivalent``
   ``"not_equivalent"``                          ``not_equivalent``
   ``"indeterminate"`` (timeout reason)          ``timeout``
   ``"indeterminate"`` (other)                   ``unknown``
   ============================================  =================

   ``smt_script`` is captured for ``not_equivalent`` / ``unknown`` /
   ``timeout`` (Req 5.5 / 5.6 / 5.9 / 6.4).

5. Apply the Expected_Fail override: store the pre-override
   ``underlying_verdict`` and the post-override ``reported_verdict`` on
   every :class:`~bird_benchmark.types.RunResult`. A non-equivalent
   underlying verdict for a listed Test_Case is reported as
   ``Verdict.expected_fail`` (Req 12.3). Listed Test_Cases whose
   underlying verdict is ``equivalent`` keep ``reported_verdict ==
   equivalent`` so the suite driver can warn about the stale entry
   (Req 12.4).

The ``reason`` field is truncated to 2000 characters at the call site
(Req 3.1 / 3.2 / 3.3 / 3.5); empty planner messages or codes are
substituted with the literal ``"no message"`` / ``"no code"``
(Req 3.3).

Import boundary
---------------

Per Req 10.4, the framework imports from ``text_to_sql_planner`` only
through public symbols. This module imports ``main.run``,
``equivalence.check_equivalence``, ``equivalence.EquivalenceCheckerConfig``
(all enumerated in the allow-list), ``equivalence.convert_to_smt``
(needed to render the captured ``smt_script``; see ``_build_smt_script``
below), ``converter.table_converter.convert_tables`` (used by
:func:`_schema_types_from_test_case` to derive a ``{column: "String"}``
map for SMT type unification — see "Schema-driven SMT typing" below),
the public DRC type ``DRCExpression`` and the public helper
``query_inner_drc`` from ``types.drc``. The :class:`EquivalenceResult`
union members are *not* imported: the runner dispatches on the public
``result.status`` literal field instead, so the result-class identities
stay an implementation detail of the equivalence checker.

Schema-driven SMT typing
------------------------

When the equivalence checker compares the generated DRC against the
gold DRC, both sides reference the same predicate symbols (the
table names) but each side's column-type inference is local to its
own DRC. If the same column appears in both DRCs but with different
inferred sorts — e.g. one side constrains it to a string literal and
the other doesn't — the merged equivalence script declares
``transactions_1k`` once and emits ``Int``/``String`` argument
constants that disagree with that signature, and cvc5 exits with a
sort error before solving.

To prevent this, the runner derives a ``{column_name: "String"}``
dictionary from the per-Test_Case ``CREATE TABLE`` schema (the same
parser the planner uses, so the typing rule can't drift) and threads
it through to the equivalence checker. The checker uses it to override
local sort inference with the schema's declared types, so both DRCs
emit constants whose sorts match the merged predicate signature.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, TextIO

from text_to_sql_planner.equivalence import (
    EquivalenceCheckerConfig,
    check_equivalence as _default_check_equivalence,
    convert_to_smt,
)
from text_to_sql_planner.converter.table_converter import (
    TableConversionFailure,
    TableConversionSuccess,
    convert_tables,
)
from text_to_sql_planner.main import (
    TextToSQLFailure,
    TextToSQLSuccess,
    run as _default_planner_run,
)
from text_to_sql_planner.types.drc import DRCExpression, query_inner_drc

from bird_benchmark.evidence import forward_call
from bird_benchmark.exec_eq import compare_executions
from bird_benchmark.expected_fail import (
    ExpectedFailLoadError,
    detect_stale,
    load as _load_expected_fail,
)
from bird_benchmark.loader import BirdLoader, BirdLoaderConfig
from bird_benchmark.manifest import Manifest, ManifestParseError
from bird_benchmark.report import (
    write_json_report as _default_write_json_report,
    write_markdown_report as _default_write_markdown_report,
)
from bird_benchmark.sql_to_drc import convert_sql as _default_convert_sql
from bird_benchmark.types import (
    ConverterError,
    ExecutionResult,
    ExecutionStatus,
    RunOptions,
    RunResult,
    SampleSummary,
    SingleSelector,
    SkippedTestCase,
    SuiteSummary,
    TestCase,
    Verdict,
)


# Length cap for ``RunResult.reason`` (Req 3.1 / 3.2 / 3.3 / 3.5). The
# spec is "at most 2000 characters"; we slice with the bound applied at
# every assignment site so the field never escapes truncation, even
# when assembled from multiple sources.
_REASON_MAX = 2000

# Substrings that mark an ``IndeterminateResult.reason`` as a timeout.
# The equivalence checker emits ``"Timeout waiting for cvc5 after Ns"``
# today; we tolerate variations like ``"timed out"`` so a future change
# to the wording does not silently demote timeouts to ``unknown``.
_TIMEOUT_REASON_MARKERS: tuple[str, ...] = ("timeout", "timed out")


def _truncate_reason(text: str) -> str:
    """Cap ``text`` at the reason-length bound (Req 3.1 / 3.2 / 3.3 / 3.5)."""

    return text[:_REASON_MAX]


def _safe_message(value: Any) -> str:
    """Return a non-empty message string, substituting ``"no message"``.

    Empty / null planner messages are reported as the literal text
    ``"no message"`` per Req 3.3 so report consumers never see a blank
    reason field.
    """

    if value is None:
        return "no message"
    text = str(value)
    return text if text else "no message"


def _safe_code(value: Any) -> str:
    """Return a non-empty error-code string, substituting ``"no code"``.

    Accepts an :class:`enum.Enum` (whose ``.value`` is used), a string,
    or any object with a useful ``str()``. Empty / null codes are
    reported as the literal text ``"no code"`` per Req 3.3.
    """

    if value is None:
        return "no code"
    if hasattr(value, "value"):
        text = str(value.value)
    else:
        text = str(value)
    return text if text else "no code"


def _is_timeout_indeterminate(reason: str) -> bool:
    """Classify an ``IndeterminateResult.reason`` as a cvc5 timeout.

    Used by the verdict mapper to distinguish ``timeout`` from generic
    ``unknown`` (Req 5.9 / 6.4). The check is case-insensitive and looks
    for any of :data:`_TIMEOUT_REASON_MARKERS` as a substring; this is
    deliberately permissive because the equivalence checker's reason
    string is a free-form English sentence rather than a machine code.
    """

    lowered = (reason or "").lower()
    return any(marker in lowered for marker in _TIMEOUT_REASON_MARKERS)


def _build_smt_script(
    generated: DRCExpression, gold: DRCExpression
) -> str:
    """Render a representative SMT-LIB rendering of both DRC sides.

    The equivalence checker writes its full equivalence script to
    stdout but does not return it, so the runner cannot capture the
    exact text cvc5 received. Instead we render the two DRC conditions
    side-by-side using the public
    :func:`text_to_sql_planner.equivalence.convert_to_smt`. The result
    is two complete SMT-LIB scripts (each with its own ``(check-sat)``)
    separated by comment banners, which is what investigators want when
    diagnosing a ``not_equivalent`` / ``unknown`` / ``timeout`` verdict
    from the markdown report (Req 9.3).

    Failures rendering either side are caught and emitted as comments
    so a malformed DRC condition cannot turn a Test_Case verdict into a
    runner exception. The runner still records the verdict, the script
    just becomes self-describing about the rendering failure.
    """

    try:
        gen_smt = convert_to_smt(generated.condition)
    except Exception as exc:  # noqa: BLE001 - never let SMT rendering crash a Test_Case
        gen_smt = f";; failed to render generated DRC: {exc}"
    try:
        gold_smt = convert_to_smt(gold.condition)
    except Exception as exc:  # noqa: BLE001 - never let SMT rendering crash a Test_Case
        gold_smt = f";; failed to render gold DRC: {exc}"
    return (
        ";; --- Generated SQL DRC condition ---\n"
        f"{gen_smt}\n\n"
        ";; --- Gold SQL DRC condition ---\n"
        f"{gold_smt}"
    )


def _result_variable_key(var: Any) -> tuple[str, ...]:
    """Return a canonical equality key for a DRC result variable.

    Two result variables compare equal iff this helper returns the
    same tuple for both. The shape is:

    * ``("column", name)`` for a :class:`ColumnVariable`.
    * ``("aggregate", FUNCTION, column)`` for an :class:`AggregateVariable`.

    Used by :func:`_truncate_to_gold_arity` to decide whether the
    generated DRC's result variables strictly extend the gold's.
    """

    var_type = getattr(var, "type", None)
    if var_type == "column":
        return ("column", str(getattr(var, "name", "")))
    if var_type == "aggregate":
        return (
            "aggregate",
            str(getattr(var, "function", "")),
            str(getattr(var, "column", "")),
        )
    # Fallback: opaque per-instance key so unknown variable shapes can
    # never be claimed as a prefix match — they'll fall through to the
    # original arity-mismatch verdict, which is the safe default.
    return ("opaque", repr(var))


def _truncate_to_gold_arity(
    generated: DRCExpression, gold: DRCExpression
) -> DRCExpression | None:
    """Return a copy of ``generated`` projected to the gold's arity, or
    ``None`` if a safe truncation isn't possible.

    Motivation
    ----------

    BIRD's gold SQL for ranking-style questions ("which gas station has
    the highest revenue") often projects only the entity (``GasStationID``)
    while the planner — answering the same question — projects both
    the entity and the ranking column (``GasStationID, SUM(Price)``).
    The two queries pick the same row but have different arities, and
    the equivalence checker correctly short-circuits on arity mismatch.

    This helper enables a narrow, opt-in tolerance: when the
    generated DRC's result variables strictly extend the gold's
    (same variables in the same order, plus extra trailing ones), we
    drop the trailing extras and retry the equivalence check on
    the truncated DRC.

    Returns
    -------
    DRCExpression | None
        A copy of ``generated`` with its ``result_variables`` truncated
        to ``len(gold.result_variables)``, when:

        * the generated DRC has *more* result variables than the gold;
        * the gold's variables match the leading prefix of the
          generated's, position-by-position, by name and aggregate
          shape.

        ``None`` otherwise — including the equal-arity case (which the
        normal equivalence check already handles) and the case where
        the generated has *fewer* variables than the gold (loss of
        information, never a safe truncation).

    Why this is a tolerance and not a proof
    ----------------------------------------

    Truncating the result variables doesn't change the underlying
    relation — the DRC's ``condition`` is unchanged, only the
    projection narrows. So if the truncated generated DRC is
    equivalent to the gold DRC, the *answer set* of the un-truncated
    query, projected to the gold's columns, is exactly the gold's
    answer set. That is the property BIRD's gold queries actually test
    (the suite compares row-sets, not SELECT lists), so accepting this
    case as ``equivalent`` matches BIRD's evaluation contract without
    weakening cvc5's role.
    """

    gen_vars = list(getattr(generated, "result_variables", []) or [])
    gold_vars = list(getattr(gold, "result_variables", []) or [])

    if len(gen_vars) <= len(gold_vars):
        # Equal arity → the normal equivalence path handles it.
        # Generated < gold → truncating would have to *invent* columns,
        # which we never do.
        return None

    # Position-by-position prefix match, by canonical key.
    for index, gold_var in enumerate(gold_vars):
        if _result_variable_key(gen_vars[index]) != _result_variable_key(gold_var):
            return None

    # Build the truncated DRC. We replace only the result_variables
    # field; the condition, and any other fields a future schema
    # extension adds, are preserved verbatim.
    truncated = DRCExpression(
        result_variables=gen_vars[: len(gold_vars)],
        condition=generated.condition,
    )
    return truncated


def _schema_types_from_test_case(test_case: TestCase) -> dict[str, str]:
    """Return a ``{column_name: "String"}`` dict for ``test_case.schema``.

    The equivalence checker accepts a ``schema_types`` dict that
    overrides per-DRC sort inference with the schema's declared
    types (only ``"String"`` overrides matter — anything else
    defaults to ``Int`` in the SMT layer). Without this, the
    generated DRC and the gold DRC can locally infer different
    sorts for the same column position; the merged equivalence
    script then declares the table predicate once with one
    signature but the two sides emit constants with mismatched
    sorts, and cvc5 rejects the script before solving.

    The runner derives the dict from ``test_case.schema`` (the
    joined ``CREATE TABLE`` statements the loader already
    produced) using the same parser the planner uses so the
    typing rule can't drift between the two pipelines. A schema
    that fails to parse yields an empty dict; the checker then
    falls back to per-DRC sort inference, which is the
    pre-existing behaviour and is correct for any Test_Case
    where both sides happen to agree on column sorts.
    """

    if not test_case.schema:
        return {}
    result = convert_tables(test_case.schema)
    if isinstance(result, TableConversionFailure):
        return {}
    if not isinstance(result, TableConversionSuccess):
        return {}
    schema_types: dict[str, str] = {}
    for relation in result.relations:
        for column, declared_type in relation.column_types.items():
            # Only ``"String"`` overrides matter to the equivalence
            # checker; ``"Int"`` is the default. If the same column
            # name appears in two tables with different declared
            # types, the union takes ``"String"`` so the checker
            # never under-declares a string column.
            if declared_type == "String":
                schema_types[column] = "String"
    return schema_types


def _fk_axioms_from_test_case(test_case: TestCase) -> list[str]:
    """Render BIRD's declared foreign keys as SMT-LIB ``(assert (forall ...))``.

    For each declared FK ``T1.c -> T2.k`` we emit the axiom

    .. code-block:: smt2

        (assert (forall ((x_T1_col_0 SortT1_0) ... (x_T1_col_N SortT1_N))
          (=> (T1 x_T1_col_0 ... x_T1_col_N)
              (exists ((y_T2_col_j SortT2_j) ...)
                (T2 y_T2_col_0 ... y_T2_col_M)))))

    where the existentially-quantified ``y_T2_col_*`` covers every
    position of ``T2`` *except* the FK position, which is replaced
    by ``x_T1_col_<from>`` so the FK constraint is what links the
    two predicate calls.

    The axiom states "every row of ``T1`` has a matching row in
    ``T2``" — exactly the referential-integrity guarantee that
    SQL's ``REFERENCES`` clause promises but the DRC condition
    doesn't otherwise restate. With it in scope, cvc5 can prove
    that ``SELECT ... FROM T1 INNER JOIN T2 ON T1.c = T2.k`` is
    equivalent to the same query without the join when the join
    contributes no extra filter, which is the BIRD gold-query
    pattern this exists to handle.

    Returns
    -------
    list[str]
        One ``(assert ...)`` line per FK. Empty when the test case
        has no FKs, or when the schema cannot be parsed (the
        equivalence proof then runs without referential-integrity
        assistance — the same fallback the runner already has for
        missing schema types).

    Per-position sorts come from the schema (parsed via the same
    ``convert_tables`` helper :func:`_schema_types_from_test_case`
    uses); columns absent from the schema map default to ``Int``,
    matching the equivalence checker's own default.
    """

    if not test_case.foreign_keys or not test_case.schema:
        return []
    parsed = convert_tables(test_case.schema)
    if not isinstance(parsed, TableConversionSuccess):
        return []

    # Index relations by lowercased table name so the FK metadata's
    # casing (which BIRD pulls from sqlite_master and may disagree
    # with the planner's parsed schema) doesn't cost us a match.
    relations_by_name: dict[str, Any] = {
        r.table_name.lower(): r for r in parsed.relations
    }

    axioms: list[str] = []
    for fk in test_case.foreign_keys:
        from_rel = relations_by_name.get(fk.from_table.lower())
        to_rel = relations_by_name.get(fk.to_table.lower())
        if from_rel is None or to_rel is None:
            continue

        # Locate the FK column positions, case-insensitively.
        from_lower = [c.lower() for c in from_rel.columns]
        to_lower = [c.lower() for c in to_rel.columns]
        try:
            from_idx = from_lower.index(fk.from_column.lower())
            to_idx = to_lower.index(fk.to_column.lower())
        except ValueError:
            # Schema lists the table but not the column — skip.
            continue

        axiom = _build_fk_axiom(
            from_table=from_rel.table_name,
            from_columns=from_rel.columns,
            from_types=from_rel.column_types,
            from_idx=from_idx,
            to_table=to_rel.table_name,
            to_columns=to_rel.columns,
            to_types=to_rel.column_types,
            to_idx=to_idx,
        )
        axioms.append(axiom)
    return axioms


def _build_fk_axiom(
    *,
    from_table: str,
    from_columns: list[str],
    from_types: dict[str, str],
    from_idx: int,
    to_table: str,
    to_columns: list[str],
    to_types: dict[str, str],
    to_idx: int,
) -> str:
    """Render a single FK as an SMT-LIB ``(assert (forall ...))`` axiom.

    Variable names are mangled with the table name and column name so
    the axiom can never accidentally collide with the equivalence
    script's own constants. Sorts default to ``Int`` for any column
    not in the type map, matching the equivalence checker's default.
    """

    def sort_for(types: dict[str, str], column: str) -> str:
        return types.get(column, "Int")

    # Variables for the universal: one per from_table column.
    from_vars = [f"_fk_{from_table}_{col}" for col in from_columns]
    from_bindings = " ".join(
        f"({var} {sort_for(from_types, col)})"
        for var, col in zip(from_vars, from_columns)
    )

    # Variables for the existential: one per to_table column EXCEPT
    # the FK position, which is replaced by the from-side variable
    # at ``from_idx``. This is the linkage that makes the axiom
    # actually say "matching".
    fk_var = from_vars[from_idx]
    to_args: list[str] = []
    to_existentials: list[tuple[str, str]] = []
    for j, col in enumerate(to_columns):
        if j == to_idx:
            to_args.append(fk_var)
        else:
            var = f"_fk_{to_table}_{col}"
            to_args.append(var)
            to_existentials.append((var, sort_for(to_types, col)))

    if to_existentials:
        ex_bindings = " ".join(f"({v} {s})" for v, s in to_existentials)
        consequent = (
            f"(exists ({ex_bindings}) ({to_table} {' '.join(to_args)}))"
        )
    else:
        consequent = f"({to_table} {' '.join(to_args)})"

    antecedent = f"({from_table} {' '.join(from_vars)})"
    return (
        f"(assert (forall ({from_bindings}) "
        f"(=> {antecedent} {consequent})))"
    )


def _apply_expected_fail(
    result: RunResult, expected_fail: set[str]
) -> RunResult:
    """Apply the Expected_Fail override to ``result``.

    The list is a one-way mask (Req 12.3 / 12.4): a non-equivalent
    underlying verdict for a listed Test_Case is *reported* as
    ``Verdict.expected_fail``, while an equivalent underlying verdict
    keeps ``reported_verdict == equivalent`` so the suite driver can
    flag the stale entry (Req 12.4). Both verdicts are stored on the
    :class:`RunResult` (Req 12.5); this helper mutates only the
    reported side.
    """

    if result.test_case_id in expected_fail:
        if result.underlying_verdict != Verdict.equivalent:
            result.reported_verdict = Verdict.expected_fail
        # else: leave reported_verdict == equivalent. The suite driver
        # walks Run_Results after the run and warns on each listed
        # Test_Case whose underlying verdict was equivalent (Req 12.4).
    return result


def _sqlite_path_for(test_case: TestCase, options: RunOptions) -> Path:
    """Build the BIRD per-database SQLite path.

    Mirrors ``BirdLoader._sqlite_path`` but takes the inputs the
    runner already has (the test case and the options) so we don't
    pass the loader through every dispatch frame. The convention is

        {bird_root}/{split}/{split}_databases/{db_id}/{db_id}.sqlite

    The runner's exec-eq step is responsible for handling a missing
    file gracefully — :func:`bird_benchmark.exec_eq.compare_executions`
    returns ``ExecutionStatus.db_unavailable`` in that case rather
    than raising.
    """

    split = options.split
    return (
        options.bird_root
        / split
        / f"{split}_databases"
        / test_case.db_id
        / f"{test_case.db_id}.sqlite"
    )


def _finalize_with_execution(
    test_case: TestCase,
    options: RunOptions,
    result: RunResult,
    *,
    exec_callable=None,
) -> RunResult:
    """Populate ``result.execution`` and return ``result``.

    Called from every exit path of :func:`run_one` so the executed-
    row-set verdict is recorded alongside the cvc5 verdict. Skips
    cleanly when:

    * ``options.execution_check`` is ``False`` (operator opted out).
    * The result has no ``generated_sql`` (planner / converter
      failure paths). The exec status is set to ``skipped`` so the
      report can show "no exec data" instead of leaving the field
      ``None``.

    The default executor is :func:`compare_executions`; tests can
    inject a fake via the ``exec_callable`` keyword so neither
    the real SQLite database nor the real disk is touched.
    """

    if not options.execution_check:
        return result

    if not result.generated_sql or not result.gold_sql:
        result.execution = ExecutionResult(
            status=ExecutionStatus.skipped,
            error="empty SQL on one or both sides",
        )
        return result

    callable_ = exec_callable if exec_callable is not None else compare_executions
    sqlite_path = _sqlite_path_for(test_case, options)
    try:
        result.execution = callable_(
            sqlite_path=sqlite_path,
            generated_sql=result.generated_sql,
            gold_sql=result.gold_sql,
            timeout_seconds=float(options.execution_timeout_seconds),
        )
    except Exception as exc:  # noqa: BLE001 - exec eq must never crash run_one
        # ``compare_executions`` is designed to return error statuses
        # rather than raise, but we belt-and-brace here so a bug in the
        # exec layer can't bring down the whole suite.
        result.execution = ExecutionResult(
            status=ExecutionStatus.db_unavailable,
            error=f"{type(exc).__name__}: {exc}",
        )
    return result


async def run_one(
    test_case: TestCase,
    options: RunOptions,
    expected_fail: set[str],
    *,
    planner_callable: Callable[..., Awaitable[Any]] = _default_planner_run,
    convert_sql_callable: Callable[[str, str], Any] = _default_convert_sql,
    check_equivalence_callable: Callable[..., Awaitable[Any]] = _default_check_equivalence,
    exec_callable: Callable[..., ExecutionResult] | None = None,
) -> RunResult:
    """Run one ``TestCase`` end-to-end and return a ``RunResult``.

    Parameters
    ----------
    test_case:
        The BIRD record to evaluate. The framework assumes the loader
        has already filled in ``schema``, ``question``, ``evidence``,
        and ``gold_sql``.
    options:
        User-supplied configuration. ``per_test_timeout_seconds`` bounds
        the planner; ``cvc5_timeout_seconds`` and ``cvc5_path`` flow
        through to ``EquivalenceCheckerConfig``.
    expected_fail:
        The Expected_Fail_List as a set of Test_Case_IDs. Empty when no
        ``--expected-fail`` was supplied. The override is applied
        in-place on the returned :class:`RunResult`.

    Keyword-only injection points
    -----------------------------
    ``planner_callable``, ``convert_sql_callable``, and
    ``check_equivalence_callable`` default to the production callables
    but can be overridden for unit and property tests so neither
    Bedrock nor cvc5 is invoked. The defaults are bound at module
    import time, not at call time, so a monkeypatch of
    ``bird_benchmark.runner._default_planner_run`` after import would
    *not* take effect — pass an explicit callable instead.

    Returns
    -------
    RunResult
        Always populated. Both ``underlying_verdict`` and
        ``reported_verdict`` are set; ``smt_script`` is non-``None``
        iff the underlying verdict is in
        ``{not_equivalent, unknown, timeout}`` (Property 14).
    """

    # --- Step 1: Build the planner call --------------------------------
    # ``forward_call`` inspects the planner signature for a dedicated
    # evidence keyword parameter and routes the BIRD evidence string
    # through it when found, or concatenates with ``\n`` otherwise. The
    # returned ``planner_input`` is the *exact* string passed as the
    # ``question`` argument so we can record it byte-for-byte in
    # ``RunResult.planner_input`` per Req 11.4.
    awaitable, planner_input = forward_call(
        question=test_case.question,
        evidence=test_case.evidence,
        run_callable=planner_callable,
        schema=test_case.schema,
    )

    # ``base`` carries the fields populated identically across every
    # exit path so each branch only fills in the verdict / reason /
    # SMT-script fields that branch is responsible for.
    base = dict(
        test_case_id=test_case.test_case_id,
        question=test_case.question,
        evidence=test_case.evidence,
        planner_input=planner_input,
        gold_sql=test_case.gold_sql,
    )

    # --- Step 2: Await the planner under the per-Test_Case budget -----
    try:
        planner_result = await asyncio.wait_for(
            awaitable, timeout=options.per_test_timeout_seconds
        )
    except asyncio.TimeoutError:
        # Req 6.6: the per-Test_Case wall-clock budget elapsed while the
        # planner was running. Record planner_failed with the literal
        # reason ``planner_timeout`` so report consumers can filter
        # timeouts from other planner failures.
        return _finalize_with_execution(
            test_case,
            options,
            _apply_expected_fail(
                RunResult(
                    **base,
                    underlying_verdict=Verdict.planner_failed,
                    reported_verdict=Verdict.planner_failed,
                    reason=_truncate_reason("planner_timeout"),
                    error_code="planner_timeout",
                ),
                expected_fail,
            ),
            exec_callable=exec_callable,
        )
    except Exception as exc:  # noqa: BLE001 - per-Test_Case isolation
        # The planner raised an unexpected exception (network glitch,
        # internal bug, etc.). Per Req 8.6 the suite must keep running;
        # the unhandled-exception → planner_failed mapping at the suite
        # level lives in the suite driver, but ``run_one`` already
        # short-circuits to a structured failure here so the suite
        # driver's catch-all is only the safety net.
        code = type(exc).__name__
        message = _safe_message(str(exc))
        return _finalize_with_execution(
            test_case,
            options,
            _apply_expected_fail(
                RunResult(
                    **base,
                    underlying_verdict=Verdict.planner_failed,
                    reported_verdict=Verdict.planner_failed,
                    reason=_truncate_reason(f"{code}: {message}"),
                    error_code=code,
                ),
                expected_fail,
            ),
            exec_callable=exec_callable,
        )

    if isinstance(planner_result, TextToSQLFailure):
        # Req 3.3 / 5.7: the planner returned a structured failure.
        # Surface both the message and the code in ``reason`` so
        # operators do not have to cross-reference the JSON report's
        # ``error_code`` field to make sense of the markdown.
        message = _safe_message(planner_result.error)
        code = _safe_code(planner_result.code)
        return _finalize_with_execution(
            test_case,
            options,
            _apply_expected_fail(
                RunResult(
                    **base,
                    underlying_verdict=Verdict.planner_failed,
                    reported_verdict=Verdict.planner_failed,
                    reason=_truncate_reason(f"{code}: {message}"),
                    error_code=code,
                ),
                expected_fail,
            ),
            exec_callable=exec_callable,
        )

    if not isinstance(planner_result, TextToSQLSuccess):
        # Defensive: the planner contract is the
        # ``TextToSQLSuccess | TextToSQLFailure`` union. Anything else
        # is a planner bug; treat it as a planner_failed so the suite
        # keeps running and the reason names the offending type.
        code = "UNEXPECTED_PLANNER_RESULT"
        message = _safe_message(
            f"unexpected planner result type: {type(planner_result).__name__}"
        )
        return _finalize_with_execution(
            test_case,
            options,
            _apply_expected_fail(
                RunResult(
                    **base,
                    underlying_verdict=Verdict.planner_failed,
                    reported_verdict=Verdict.planner_failed,
                    reason=_truncate_reason(f"{code}: {message}"),
                    error_code=code,
                ),
                expected_fail,
            ),
            exec_callable=exec_callable,
        )

    generated_sql = planner_result.sql or ""
    generated_drc = planner_result.target_expression
    base["generated_sql"] = generated_sql

    # --- Step 3: Convert the gold SQL ---------------------------------
    # ``convert_sql`` returns either a ``QueryExpression`` or a
    # ``ConverterError``; it never raises (see its docstring). Map the
    # two error kinds to the verdicts spelled out in the task:
    # ``unsupported_feature`` → ``gold_conversion_failure`` (Req 5.8),
    # ``parse_error`` → ``skipped`` (Req 3.2). ``unbound_reference`` is
    # a translator-side failure that semantically belongs with
    # ``gold_conversion_failure`` (the gold is internally consistent
    # SQL but the converter cannot bind one of its references).
    gold_query = convert_sql_callable(test_case.gold_sql, test_case.schema)
    if isinstance(gold_query, ConverterError):
        if gold_query.kind == "parse_error":
            verdict = Verdict.skipped
        else:
            # ``unsupported_feature`` and ``unbound_reference`` both
            # mean "we cannot evaluate this gold against the planner";
            # operators want them in one bucket so they can triage the
            # converter.
            verdict = Verdict.gold_conversion_failure
        message = _safe_message(gold_query.message)
        if gold_query.feature:
            reason_text = f"{gold_query.feature}: {message}"
        else:
            reason_text = message
        return _finalize_with_execution(
            test_case,
            options,
            _apply_expected_fail(
                RunResult(
                    **base,
                    underlying_verdict=verdict,
                    reported_verdict=verdict,
                    reason=_truncate_reason(reason_text),
                ),
                expected_fail,
            ),
            exec_callable=exec_callable,
        )

    # ``convert_sql`` may return a ``QueryExpression`` wrapped in
    # ``LimitExpression`` / ``OrderByExpression``; the equivalence
    # checker only operates on the inner core DRC, so peel the
    # wrappers off via the public helper.
    try:
        gold_drc = query_inner_drc(gold_query)
    except TypeError as exc:
        # ``query_inner_drc`` raises ``TypeError`` when the inner-most
        # node is not a ``DRCExpression``. Treat this as a
        # gold-conversion failure so the suite keeps running and the
        # reason names the offending type.
        return _finalize_with_execution(
            test_case,
            options,
            _apply_expected_fail(
                RunResult(
                    **base,
                    underlying_verdict=Verdict.gold_conversion_failure,
                    reported_verdict=Verdict.gold_conversion_failure,
                    reason=_truncate_reason(_safe_message(str(exc))),
                ),
                expected_fail,
            ),
            exec_callable=exec_callable,
        )

    # --- Step 4: Equivalence check -----------------------------------
    config = EquivalenceCheckerConfig(
        timeout_seconds=options.cvc5_timeout_seconds,
        cvc5_path=options.cvc5_path,
    )
    # Build the schema-types map from the BIRD ``CREATE TABLE``
    # statements so cvc5 sees one consistent column sort for each
    # predicate position across both sides of the equivalence script.
    # See "Schema-driven SMT typing" in the module docstring.
    schema_types = _schema_types_from_test_case(test_case)
    # Render BIRD's declared foreign keys into SMT-LIB axioms so the
    # equivalence proof can use referential-integrity facts. Empty
    # when the test case has no FKs or when the schema can't be
    # parsed; the equivalence call then runs unaffected.
    fk_axioms = _fk_axioms_from_test_case(test_case)
    # Surface a clear section header before the final equivalence
    # check so the run log makes the boundary obvious between
    # *planner-internal* equivalence checks (built relation vs target
    # DRC) and the *suite-level* check that decides the verdict
    # (generated DRC vs gold DRC). Without this, both checks share
    # the same ``### cvc5 equivalence check`` banner and operators
    # mis-read a final not_equivalent as a planner failure.
    print(
        "\n## BIRD final equivalence check\n\n"
        "_Comparing the planner's generated DRC against the "
        "gold DRC translated from BIRD's reference SQL._\n",
        flush=True,
    )
    eq_result = await check_equivalence_callable(
        generated_drc, gold_drc, config,
        schema_types=schema_types, axioms=fk_axioms,
        label=f"BIRD final: generated DRC vs gold DRC ({test_case.test_case_id})",
        lhs_label="generated",
        rhs_label="gold",
    )

    # ---- BIRD-style projection-tolerance retry ----------------------
    # If the equivalence check rejected the pair on arity grounds AND
    # the generated DRC's result variables strictly extend the gold's,
    # retry against a truncated copy of the generated DRC. This
    # narrowly handles BIRD's "which X has the highest Y" pattern where
    # gold projects only X but the planner projects (X, Y). The
    # condition is unchanged — only the trailing projection columns
    # are dropped — so a positive verdict here means the un-truncated
    # query's answer set, projected to the gold's columns, matches
    # the gold's answer set, which is what BIRD's evaluation actually
    # tests.
    #
    # We only retry when the original verdict was ``not_equivalent``;
    # ``equivalent`` doesn't need retrying, and ``indeterminate`` /
    # ``unknown`` already capture cvc5 fragility we don't want to
    # paper over.
    if getattr(eq_result, "status", None) == "not_equivalent":
        truncated = _truncate_to_gold_arity(generated_drc, gold_drc)
        if truncated is not None:
            retry_result = await check_equivalence_callable(
                truncated, gold_drc, config,
                schema_types=schema_types, axioms=fk_axioms,
                label=(
                    f"BIRD final retry: generated DRC truncated to gold's "
                    f"arity vs gold DRC ({test_case.test_case_id})"
                ),
                lhs_label="generated (truncated)",
                rhs_label="gold",
            )
            if getattr(retry_result, "status", None) == "equivalent":
                # The retry replaces the original verdict so downstream
                # status dispatch sees the relaxed answer.
                eq_result = retry_result

    # We dispatch on the public ``status`` literal rather than
    # ``isinstance`` against the result-class union, so this module
    # does not have to import ``EquivalentResult`` /
    # ``NotEquivalentResult`` / ``IndeterminateResult`` and the import
    # boundary stays at the symbols enumerated in Req 10.4 (modulo
    # ``convert_to_smt``, see module docstring).
    status = getattr(eq_result, "status", None)
    smt_script: str | None = None
    reason_text = ""

    if status == "equivalent":
        verdict = Verdict.equivalent
    elif status == "not_equivalent":
        verdict = Verdict.not_equivalent
        smt_script = _build_smt_script(generated_drc, gold_drc)
    elif status == "indeterminate":
        ind_reason = getattr(eq_result, "reason", "") or ""
        if _is_timeout_indeterminate(ind_reason):
            verdict = Verdict.timeout
        else:
            verdict = Verdict.unknown
        smt_script = _build_smt_script(generated_drc, gold_drc)
        reason_text = _truncate_reason(_safe_message(ind_reason))
    else:
        # Unknown status string: equivalent to a malformed equivalence
        # checker contract. Treat as ``unknown`` and capture the SMT
        # script so an operator can inspect both sides.
        verdict = Verdict.unknown
        smt_script = _build_smt_script(generated_drc, gold_drc)
        reason_text = _truncate_reason(
            _safe_message(
                f"unexpected EquivalenceResult: {type(eq_result).__name__}"
                f" (status={status!r})"
            )
        )

    return _finalize_with_execution(
        test_case,
        options,
        _apply_expected_fail(
            RunResult(
                **base,
                underlying_verdict=verdict,
                reported_verdict=verdict,
                smt_script=smt_script,
                reason=reason_text,
            ),
            expected_fail,
        ),
        exec_callable=exec_callable,
    )


__all__ = [
    "run_one", "run_single", "run_suite", "run_sample", "SingleSelectorError",
]


# =====================================================================
# Single_Run driver
# =====================================================================


class SingleSelectorError(Exception):
    """Raised by :func:`run_single` when the selector cannot resolve to a
    unique :class:`TestCase`.

    The CLI layer (task 14.1) catches this and exits non-zero with the
    error's message attached to stderr; tests assert on the message text
    rather than the exit code so the wiring stays decoupled from
    ``sys.exit``.

    Four shapes use this exception (Req 7.4 / 7.5 / 7.6):

    * Neither ``test_case_id`` nor ``question_text`` was supplied.
    * Both selectors were supplied at once.
    * The selector matched zero Test_Cases in the requested split.
    * The ``question_text`` selector matched two or more Test_Cases.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _serialize_run_result(result: RunResult) -> dict[str, Any]:
    """Return a JSON-serialisable dict for a :class:`RunResult`.

    Mirrors the shape :mod:`bird_benchmark.manifest` writes so the
    single-mode output and the suite manifest stay in lock-step. ``Verdict``
    is a ``str``-Enum, so :func:`dataclasses.asdict` already produces a
    JSON-friendly value, but we keep an enum-aware ``default`` callback
    below for any future enum field.
    """

    return asdict(result)


def _default_json_serializer(obj: Any) -> Any:
    """Fallback ``json.dumps`` serialiser for non-standard types."""

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Cannot serialise {type(obj).__name__}: {obj!r}")


def _validate_selector(selector: SingleSelector) -> None:
    """Enforce the exactly-one rule on the :class:`SingleSelector`.

    Raises :class:`SingleSelectorError` for the zero-and-both cases per
    Req 7.6. ``test_case_id`` / ``question_text`` are treated as "not
    supplied" only when ``None``; an empty string is a deliberate match
    request and is left for the filter step to reject as a zero-match
    selector.
    """

    has_id = selector.test_case_id is not None
    has_question = selector.question_text is not None
    if has_id and has_question:
        raise SingleSelectorError(
            "exactly one of --id / --question is required (got both)"
        )
    if not has_id and not has_question:
        raise SingleSelectorError(
            "exactly one of --id / --question is required (got neither)"
        )


def _resolve_selector(
    selector: SingleSelector,
    options: RunOptions,
    *,
    loader_factory: Callable[[BirdLoaderConfig], BirdLoader] = BirdLoader,
) -> TestCase:
    """Find the unique :class:`TestCase` that matches ``selector``.

    Filters case-sensitively on ``test_case_id`` or ``question`` (Req 7.1 /
    7.2). Skipped records (missing required fields, missing per-record
    SQLite files) are ignored — they cannot be the match because they have
    no question text and their identifier shape differs from the regular
    Test_Case_IDs.

    Raises :class:`SingleSelectorError` when the selector matches zero
    Test_Cases (Req 7.4) or when ``question_text`` matches two or more
    Test_Cases (Req 7.5).
    """

    loader = loader_factory(
        BirdLoaderConfig(bird_root=options.bird_root, split=options.split)
    )

    matches: list[TestCase] = []
    for item in loader.load():
        if isinstance(item, SkippedTestCase):
            # Skipped records cannot be the match: a record missing a
            # required field has no usable question_text or gold SQL, and
            # its synthetic Test_Case_ID (e.g. ``dev_record_3``) would
            # have to be supplied verbatim — which is not the contract.
            continue

        if selector.test_case_id is not None:
            if item.test_case_id == selector.test_case_id:
                matches.append(item)
                # ``test_case_id`` is unique by construction
                # (``f"{split}_{question_id}"``) so the first match is the
                # only match. Bail early to avoid pulling the rest of the
                # split into memory.
                break
        else:
            # ``question_text`` selector. Walk the entire stream because
            # multiple Test_Cases can share question text, and we need
            # the count to report ambiguity (Req 7.5).
            if item.question == selector.question_text:
                matches.append(item)

    if not matches:
        if selector.test_case_id is not None:
            label = f"test_case_id={selector.test_case_id!r}"
        else:
            label = f"question={selector.question_text!r}"
        raise SingleSelectorError(
            f"selector {label} matched no Test_Case in split {options.split!r}"
        )

    if len(matches) > 1:
        # Only ``question_text`` can produce multiple matches; the
        # ``test_case_id`` branch breaks after the first hit.
        raise SingleSelectorError(
            f"selector question={selector.question_text!r} matched "
            f"{len(matches)} Test_Cases in split {options.split!r} "
            f"(expected exactly one)"
        )

    return matches[0]


async def run_single(
    selector: SingleSelector,
    options: RunOptions,
    *,
    expected_fail: set[str] | None = None,
    stdout: TextIO | None = None,
    planner_callable: Callable[..., Awaitable[Any]] = _default_planner_run,
    convert_sql_callable: Callable[[str, str], Any] = _default_convert_sql,
    check_equivalence_callable: Callable[..., Awaitable[Any]] = _default_check_equivalence,
    exec_callable: Callable[..., ExecutionResult] | None = None,
    loader_factory: Callable[[BirdLoaderConfig], BirdLoader] = BirdLoader,
) -> RunResult:
    """Run exactly one :class:`TestCase` selected by ``selector``.

    Validates the selector, loads the requested split, finds the unique
    Test_Case match, runs :func:`run_one`, and prints the resulting
    :class:`RunResult` as a JSON object to ``stdout`` (default
    :data:`sys.stdout`). Returns the :class:`RunResult` so library callers
    can inspect it without parsing stdout.

    The four selector-error cases all raise :class:`SingleSelectorError`
    so the CLI can map them to a non-zero exit (Req 7.4 / 7.5 / 7.6); the
    exact CLI exit-code wiring lives in task 14.1.

    Parameters
    ----------
    selector:
        Exactly one of ``test_case_id`` / ``question_text`` must be set.
    options:
        Same configuration object used by :func:`run_suite`. The
        ``bird_root`` / ``split`` fields drive the loader; the timeout
        and cvc5 fields flow through :func:`run_one`.
    expected_fail:
        Optional pre-loaded Expected_Fail_List. When ``None``, the list
        is loaded from ``options.expected_fail_path`` so single-mode
        respects the same override as suite-mode (Req 12.3).
    stdout:
        Output stream for the JSON record. Defaults to :data:`sys.stdout`;
        tests can pass a :class:`io.StringIO` to capture without
        monkeypatching.

    Keyword-only callables and ``loader_factory`` mirror :func:`run_one`
    so unit and property tests can substitute deterministic fakes for
    Bedrock / cvc5 / the BIRD download.
    """

    _validate_selector(selector)

    if expected_fail is None:
        expected_fail = _load_expected_fail(options.expected_fail_path)

    test_case = _resolve_selector(
        selector, options, loader_factory=loader_factory
    )

    out = stdout if stdout is not None else sys.stdout

    _write_bird_test_case_banner(out, test_case)

    result = await run_one(
        test_case,
        options,
        expected_fail,
        planner_callable=planner_callable,
        convert_sql_callable=convert_sql_callable,
        check_equivalence_callable=check_equivalence_callable,
        exec_callable=exec_callable,
    )

    _write_run_result_block(out, result)

    return result


def _write_bird_test_case_banner(out: TextIO, test_case: TestCase) -> None:
    """Print the ``# BIRD Test_Case`` header block for ``test_case``.

    Surfaces the test_case_id, db_id, question, and gold SQL above
    the planner / cvc5 transcripts so an operator scrolling through
    the run log sees what the suite is comparing against without
    having to scroll past the question converter's prompt.
    """

    out.write("\n# BIRD Test_Case\n\n")
    out.write(f"- **Test_Case_ID:** `{test_case.test_case_id}`\n")
    out.write(f"- **Database:** `{test_case.db_id}`\n")
    out.write(f"- **Question:** {test_case.question}\n")
    if test_case.evidence:
        out.write(f"- **Evidence:** {test_case.evidence}\n")
    out.write("\n**Gold SQL (from BIRD):**\n\n")
    out.write("```sql\n")
    out.write(test_case.gold_sql.rstrip())
    out.write("\n```\n\n")
    out.flush()


def _write_run_result_block(out: TextIO, result: RunResult) -> None:
    """Print the ``# Run_Result`` fenced JSON block for ``result``.

    The JSON is indented for human readability — ``json.loads`` round
    trips indented JSON fine, so machine consumers (tests, scripts)
    are unaffected.
    """

    payload = _serialize_run_result(result)
    out.write("\n# Run_Result\n\n```json\n")
    out.write(
        json.dumps(payload, default=_default_json_serializer, indent=2)
    )
    out.write("\n```\n")
    out.flush()


# =====================================================================
# Suite_Run driver
# =====================================================================


# Default report paths used when ``RunOptions.report_json_path`` /
# ``RunOptions.report_md_path`` are left unset. The CLI (task 14.1)
# wires its own defaults; library callers fall back to writing the
# reports next to the working directory so a programmatic ``run_suite``
# call always produces both reports.
_DEFAULT_REPORT_JSON_PATH = Path("bird-report.json")
_DEFAULT_REPORT_MD_PATH = Path("bird-report.md")


def _make_skipped_run_result(skipped: SkippedTestCase) -> RunResult:
    """Turn a :class:`SkippedTestCase` into a :class:`RunResult`.

    Loader skips have no question / evidence / gold SQL because the
    record was malformed; we record the empty strings explicitly so the
    reporter does not have to special-case ``None`` values, and we
    surface the skip reason in ``RunResult.reason`` (truncated, like
    every other reason field, per Req 3.x).
    """

    return RunResult(
        test_case_id=skipped.test_case_id,
        underlying_verdict=Verdict.skipped,
        reported_verdict=Verdict.skipped,
        question="",
        evidence="",
        planner_input="",
        generated_sql="",
        gold_sql="",
        smt_script=None,
        reason=_truncate_reason(_safe_message(skipped.reason)),
        error_code="",
        execution=ExecutionResult(
            status=ExecutionStatus.skipped,
            error="loader skipped this record",
        ),
    )


def _make_unhandled_exception_run_result(
    test_case: TestCase, exc: BaseException
) -> RunResult:
    """Convert an unhandled exception from ``run_one`` into a Run_Result.

    Per Req 8.6, the suite driver must keep going when a single
    Test_Case raises. The recorded verdict is :attr:`Verdict.planner_failed`
    (the design's catch-all for "the planner side of the pipeline did
    not produce a usable answer") and the reason names the exception
    type and message so an operator can triage from the report alone.
    """

    code = type(exc).__name__
    message = _safe_message(str(exc))
    return RunResult(
        test_case_id=test_case.test_case_id,
        underlying_verdict=Verdict.planner_failed,
        reported_verdict=Verdict.planner_failed,
        question=test_case.question,
        evidence=test_case.evidence,
        planner_input="",
        generated_sql="",
        gold_sql=test_case.gold_sql,
        smt_script=None,
        reason=_truncate_reason(f"{code}: {message}"),
        error_code=code,
        execution=ExecutionResult(
            status=ExecutionStatus.skipped,
            error="run_one raised before producing SQL",
        ),
    )


def _emit_stale_expected_fail_warnings(
    expected_fail: set[str],
    seen_ids: set[str],
    results: list[RunResult],
    stderr: TextIO,
) -> set[str]:
    """Emit suite-end warnings for stale Expected_Fail_List entries.

    Two stale-entry shapes are reported (Req 12.4 / 12.6):

    1. **Equivalent-on-list** (Req 12.4): a Test_Case appears in the
       Expected_Fail_List but its underlying verdict was
       :attr:`Verdict.equivalent`. The list entry is now stale because
       the planner started getting it right; the operator should prune
       the entry.
    2. **Unmatched-list-IDs** (Req 12.6): a Test_Case_ID appears in the
       Expected_Fail_List but no Test_Case with that ID was seen in the
       split. Either the split changed or the list typoed an ID.

    Returns the union of the two stale-ID sets so the caller can stash
    them on :attr:`SuiteSummary.stale_expected_fail_ids`.
    """

    stale_equivalent: set[str] = {
        r.test_case_id
        for r in results
        if r.test_case_id in expected_fail
        and r.underlying_verdict == Verdict.equivalent
    }
    for tid in sorted(stale_equivalent):
        stderr.write(
            f"warning: Test_Case {tid} is on the Expected_Fail_List "
            f"but the underlying verdict was equivalent; the entry is "
            f"stale and should be removed.\n"
        )

    unmatched = detect_stale(expected_fail, seen_ids)
    for tid in sorted(unmatched):
        stderr.write(
            f"warning: Expected_Fail_List entry {tid!r} did not match "
            f"any Test_Case in the requested split.\n"
        )

    return stale_equivalent | unmatched


async def run_suite(
    options: RunOptions,
    *,
    expected_fail: set[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    planner_callable: Callable[..., Awaitable[Any]] = _default_planner_run,
    convert_sql_callable: Callable[[str, str], Any] = _default_convert_sql,
    check_equivalence_callable: Callable[..., Awaitable[Any]] = _default_check_equivalence,
    exec_callable: Callable[..., ExecutionResult] | None = None,
    loader_factory: Callable[[BirdLoaderConfig], BirdLoader] = BirdLoader,
    manifest_factory: Callable[[Path], Manifest] = Manifest.open_for_append,
    write_json_report: Callable[[list[RunResult], Path], None] = _default_write_json_report,
    write_markdown_report: Callable[[list[RunResult], Path], None] = _default_write_markdown_report,
) -> SuiteSummary:
    """Run an entire BIRD split and return a :class:`SuiteSummary`.

    Implements the design's ``run_suite`` pseudocode:

    1. Load the Expected_Fail_List from ``options.expected_fail_path``
       (Req 12.1 / 12.2). Empty when no path is supplied.
    2. Open the manifest for append, creating its parent directory when
       needed (Req 8.7). When ``options.resume`` is *false* and the
       manifest already exists, the existing file is truncated before
       the append handle is opened so a fresh suite never inherits
       entries from a prior unrelated run.
    3. When ``options.resume`` is set and the manifest exists, read
       ``completed_ids`` from it; ``Manifest.read_completed_ids`` raises
       :class:`ManifestParseError` on malformed lines, which propagates
       so the CLI can exit non-zero (Req 8.5). When ``resume`` is set
       but the manifest does not yet exist, the suite starts fresh
       (Req 8.7).
    4. Iterate the :class:`BirdLoader` stream. For each Test_Case:

       * a :class:`SkippedTestCase` becomes a synthetic
         :attr:`Verdict.skipped` Run_Result (Req 1.7);
       * a Test_Case whose ID is in ``completed_ids`` is skipped
         entirely so the suite does not duplicate work after a resume
         (Req 8.3);
       * otherwise, ``run_one`` is called inside a ``try / except``
         that converts any unhandled exception into a
         :attr:`Verdict.planner_failed` Run_Result tagged with the
         exception type and message (Req 8.6).

    5. Append every Run_Result to the manifest *before* the next
       Test_Case begins (Req 8.2), tally it on the
       :class:`SuiteSummary`, and print one progress line per Test_Case
       (``"{test_case_id} {reported_verdict}"``, Req 8.4).
    6. After the loop, emit stale-entry warnings (Req 12.4 / 12.6) and
       call ``write_json_report`` / ``write_markdown_report`` against
       the configured paths. ``OSError`` / ``IOError`` raised by either
       writer flips ``SuiteSummary.failed_to_write_report`` and emits
       an error message naming the report and the underlying error so
       the CLI (task 14.1) can map it to a non-zero exit (Req 9.6).

    Keyword-only injection points
    -----------------------------
    Every collaborator the suite driver touches has an injection point
    so suite-level property tests in tasks 12.2 / 12.3 / 12.4 can
    substitute deterministic fakes. The defaults are bound at import
    time, so monkeypatching the module attributes after import has no
    effect — pass an explicit callable instead. The pattern mirrors
    :func:`run_one` / :func:`run_single`.
    """

    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    # --- Step 1: Load Expected_Fail_List ------------------------------
    if expected_fail is None:
        expected_fail = _load_expected_fail(options.expected_fail_path)

    # --- Step 2: Open manifest for append -----------------------------
    manifest_path = options.manifest_path
    if manifest_path is None:
        # Library-level safety: the CLI requires --manifest in suite
        # mode, but a programmatic caller might forget. Refuse early
        # with a clear message rather than crashing inside the loader.
        raise ValueError(
            "RunOptions.manifest_path is required for suite mode"
        )

    # When ``--resume`` is NOT set and the manifest already exists,
    # truncate it before opening for append. Without this, entries from
    # a prior unrelated run would silently interleave with this run's
    # entries, breaking the "one Run_Result per Test_Case_ID" invariant
    # the suite relies on (Req 8.1). Resume mode preserves the existing
    # file so a re-invocation can pick up where it left off (Req 8.7).
    if not options.resume and manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("", encoding="utf-8")

    manifest = manifest_factory(manifest_path)

    # --- Step 3: Read completed IDs when --resume is set --------------
    completed_ids: set[str] = set()
    if options.resume:
        # ``read_completed_ids`` already returns an empty set for a
        # missing file, but we keep the explicit branch so a future
        # change to that contract does not silently change suite
        # behaviour. Req 8.7: --resume + no manifest -> start fresh.
        if manifest_path.exists():
            # ``ManifestParseError`` propagates so the CLI can exit
            # non-zero with the parse error attached (Req 8.5).
            completed_ids = manifest.read_completed_ids()

    # --- Step 4: Iterate the BIRD loader stream -----------------------
    summary = SuiteSummary()
    seen_ids: set[str] = set()

    loader = loader_factory(
        BirdLoaderConfig(bird_root=options.bird_root, split=options.split)
    )

    try:
        for item in loader.load():
            if isinstance(item, SkippedTestCase):
                # The loader synthesises a Test_Case_ID for malformed
                # records (e.g. ``dev_record_3``) so the manifest still
                # has a stable join key. Resume can therefore skip
                # already-recorded skipped Test_Cases just like normal
                # ones (Req 8.3).
                if item.test_case_id in completed_ids:
                    seen_ids.add(item.test_case_id)
                    continue
                seen_ids.add(item.test_case_id)
                result = _make_skipped_run_result(item)
            else:
                seen_ids.add(item.test_case_id)
                if item.test_case_id in completed_ids:
                    # Resume: this Test_Case was completed in a prior
                    # run; skip it without re-invoking the planner.
                    continue
                try:
                    result = await run_one(
                        item,
                        options,
                        expected_fail,
                        planner_callable=planner_callable,
                        convert_sql_callable=convert_sql_callable,
                        check_equivalence_callable=check_equivalence_callable,
                        exec_callable=exec_callable,
                    )
                except Exception as exc:  # noqa: BLE001 - per-Test_Case isolation
                    # Req 8.6: an unhandled exception from ``run_one``
                    # must not abort the suite. Convert it into a
                    # ``planner_failed`` Run_Result whose reason names
                    # the exception type and message, append it to the
                    # manifest, and continue.
                    result = _make_unhandled_exception_run_result(item, exc)

            # --- Step 5: Append, tally, progress -----------------------
            # The append happens before the next Test_Case begins
            # (Req 8.2); ``Manifest.append`` flushes + fsyncs so a
            # SIGKILL after this point leaves the file with a complete
            # record.
            manifest.append(result)
            summary.results.append(result)
            summary.counts[result.reported_verdict] = (
                summary.counts.get(result.reported_verdict, 0) + 1
            )
            out.write(
                f"{result.test_case_id} {result.reported_verdict.value}\n"
            )
            out.flush()
    finally:
        manifest.close()

    # --- Step 6: Stale warnings + reports ----------------------------
    summary.stale_expected_fail_ids = _emit_stale_expected_fail_warnings(
        expected_fail, seen_ids, summary.results, err
    )

    json_report_path = options.report_json_path or _DEFAULT_REPORT_JSON_PATH
    md_report_path = options.report_md_path or _DEFAULT_REPORT_MD_PATH

    # Each report is written under its own try/except so a failure on
    # one does not prevent the other from being attempted. Both go via
    # ``failed_to_write_report`` so the CLI sees a single boolean
    # whichever (or both) fail.
    try:
        write_json_report(summary.results, json_report_path)
    except (OSError, IOError) as exc:
        summary.failed_to_write_report = True
        err.write(
            f"error: failed to write JSON report to {json_report_path}: "
            f"{exc}\n"
        )
    try:
        write_markdown_report(summary.results, md_report_path)
    except (OSError, IOError) as exc:
        summary.failed_to_write_report = True
        err.write(
            f"error: failed to write markdown report to {md_report_path}: "
            f"{exc}\n"
        )

    return summary


# =====================================================================
# Sample_Run driver
# =====================================================================


async def run_sample(
    options: RunOptions,
    *,
    count: int,
    seed: int,
    output_dir: Path | None = None,
    expected_fail: set[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    planner_callable: Callable[..., Awaitable[Any]] = _default_planner_run,
    convert_sql_callable: Callable[[str, str], Any] = _default_convert_sql,
    check_equivalence_callable: Callable[..., Awaitable[Any]] = _default_check_equivalence,
    exec_callable: Callable[..., ExecutionResult] | None = None,
    loader_factory: Callable[[BirdLoaderConfig], BirdLoader] = BirdLoader,
) -> SampleSummary:
    """Run ``count`` randomly-selected Test_Cases from the split.

    Sample mode is the diagnostic verb between :func:`run_single`
    (one Test_Case) and :func:`run_suite` (the whole split): it
    exercises a deterministic random sample so an operator can get
    a quick estimate of the framework's pass rate without committing
    to the full suite. The seed is required so two runs of the same
    split + count + seed produce the same selection — without that
    you can't tell apart "my code got better" from "I sampled
    different cases this time".

    Output layout
    -------------

    When ``output_dir`` is ``None`` (the library default), the entire
    run — per-case banners, planner transcript, Run_Result JSON,
    aggregate summary — streams to ``stdout``. Useful for tests and
    notebook callers that want everything in one buffer.

    When ``output_dir`` is set (the CLI default), the layout is:

    * ``{output_dir}/{test_case_id}.md`` — full transcript per case
      (BIRD context block + planner / cvc5 transcript + Run_Result
      JSON). The transcript is captured by redirecting
      :data:`sys.stdout` for the duration of each ``run_one`` call,
      since the planner and equivalence checker print via the global
      ``print``.
    * ``{output_dir}/summary.md`` — aggregate summary with linked
      Test_Case_ID lists per verdict and per execution-status bucket.
      Each ID is a relative-link to its transcript file so the
      operator can click straight from the summary into the case.
    * ``{output_dir}/summary.json`` — machine-readable mirror.
    * ``stdout`` only gets per-case progress lines (``[i/n] ID →
      file``) and the headline rates at the end.

    Parameters
    ----------
    options:
        Same as the other entry points; ``bird_root`` and ``split``
        drive the loader, the timeouts flow through, and
        ``execution_check`` controls whether each case also runs
        through the SQLite execution-equivalence check.
    count:
        Number of Test_Cases to draw. Must be ≥ 1. When the split
        has fewer eligible records than ``count``, the sampler runs
        every eligible record once and reports the smaller actual
        count in :attr:`SampleSummary.sampled_count`.
    seed:
        PRNG seed for reproducibility. Required.
    output_dir:
        See "Output layout" above. ``None`` keeps the legacy
        single-stream behaviour.
    expected_fail / stdout / stderr / *_callable / loader_factory:
        Mirror :func:`run_suite` exactly. Tests inject the callables
        to drive the sampler without Bedrock or cvc5.

    Returns
    -------
    SampleSummary
        Tally of every cell needed for a one-screen status report:
        per-verdict counts, per-execution-status counts, multiset
        match count, and the two interesting disagreement cells
        (logical-yes/exec-no and logical-no/exec-yes).

    Raises
    ------
    ValueError
        For ``count < 1`` or ``seed < 0``. The CLI catches these and
        maps them to ``EXIT_CONFIG_ERROR``.
    """

    if count < 1:
        raise ValueError(f"--count must be a positive integer, got {count}")
    if seed < 0:
        raise ValueError(f"--seed must be a non-negative integer, got {seed}")

    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"--output-dir {output_dir} is not writable: {exc}"
            ) from exc

    if expected_fail is None:
        expected_fail = _load_expected_fail(options.expected_fail_path)

    # Load the split into memory. BIRD splits are a few thousand
    # records; holding them all is fine and lets us sample uniformly.
    loader = loader_factory(
        BirdLoaderConfig(bird_root=options.bird_root, split=options.split)
    )
    test_cases: list[TestCase] = []
    skipped_records: list[SkippedTestCase] = []
    for item in loader.load():
        if isinstance(item, SkippedTestCase):
            skipped_records.append(item)
        else:
            test_cases.append(item)

    if not test_cases:
        # Nothing to sample. Surface a non-fatal warning so the
        # operator can tell apart "the split is empty" from "the
        # planner failed every record".
        err.write(
            "warning: no eligible Test_Cases in split "
            f"{options.split!r}; sample is empty\n"
        )
        err.flush()
        return SampleSummary(
            seed=seed,
            requested_count=count,
            sampled_count=0,
        )

    # Deterministic sample. ``random.sample`` is "without replacement"
    # so each Test_Case appears at most once per run.
    import random as _random
    rng = _random.Random(seed)
    actual_count = min(count, len(test_cases))
    selected: list[TestCase] = rng.sample(test_cases, k=actual_count)

    # --- Header banner -------------------------------------------------
    if output_dir is None:
        out.write("\n# BIRD Sample_Run\n\n")
        out.write(
            f"- **Split:** `{options.split}`\n"
            f"- **Requested:** {count}\n"
            f"- **Sampled:** {actual_count}"
        )
        if actual_count < count:
            out.write(
                f"  _(only {len(test_cases)} eligible Test_Cases in split)_"
            )
        out.write("\n")
        out.write(f"- **Seed:** {seed}\n")
        out.write(
            f"- **Selected IDs:** "
            f"{', '.join('`' + c.test_case_id + '`' for c in selected)}\n\n"
        )
        out.flush()
    else:
        out.write(
            f"\nBIRD sample_run (split={options.split}, "
            f"count={actual_count}/{count}, seed={seed})\n"
            f"Output dir: {output_dir}\n\n"
        )
        out.flush()

    summary = SampleSummary(
        seed=seed,
        requested_count=count,
        sampled_count=actual_count,
    )

    # --- Per-Test_Case loop -------------------------------------------
    for index, test_case in enumerate(selected, start=1):
        if output_dir is None:
            # Legacy single-stream output: stream the transcript
            # straight to ``out``.
            out.write(f"\n---\n## Sample {index}/{actual_count}\n")
            out.flush()
            _write_bird_test_case_banner(out, test_case)
            try:
                result = await run_one(
                    test_case,
                    options,
                    expected_fail,
                    planner_callable=planner_callable,
                    convert_sql_callable=convert_sql_callable,
                    check_equivalence_callable=check_equivalence_callable,
                    exec_callable=exec_callable,
                )
            except Exception as exc:  # noqa: BLE001 - per-Test_Case isolation
                result = _make_unhandled_exception_run_result(test_case, exc)
            _write_run_result_block(out, result)
        else:
            # Per-case file mode: capture the entire transcript into
            # ``{output_dir}/{test_case_id}.md`` by redirecting
            # ``sys.stdout`` for the duration of ``run_one``. The
            # planner / cvc5 / question-converter all print via the
            # module-level ``print``, so this is the principled way to
            # peel them off the global stream without rewiring every
            # caller.
            case_path = output_dir / f"{test_case.test_case_id}.md"
            try:
                with case_path.open("w", encoding="utf-8") as case_file:
                    with contextlib.redirect_stdout(case_file):
                        _write_bird_test_case_banner(case_file, test_case)
                        try:
                            result = await run_one(
                                test_case,
                                options,
                                expected_fail,
                                planner_callable=planner_callable,
                                convert_sql_callable=convert_sql_callable,
                                check_equivalence_callable=check_equivalence_callable,
                                exec_callable=exec_callable,
                            )
                        except Exception as exc:  # noqa: BLE001 - per-Test_Case isolation
                            result = _make_unhandled_exception_run_result(test_case, exc)
                        _write_run_result_block(case_file, result)
            except OSError as exc:
                # Per-case file write failures shouldn't abort the
                # whole sample. Record a synthetic planner-failed
                # result so the summary still tallies the case, and
                # surface the path on stderr.
                err.write(
                    f"error: failed to write per-case transcript "
                    f"{case_path}: {exc}\n"
                )
                err.flush()
                result = _make_unhandled_exception_run_result(test_case, exc)
            # Per-case progress line on stdout — short, greppable.
            verdict_str = result.reported_verdict.value
            exec_str = (
                result.execution.status.value
                if result.execution is not None
                else "n/a"
            )
            out.write(
                f"[{index}/{actual_count}] {test_case.test_case_id} "
                f"verdict={verdict_str} exec={exec_str} → {case_path}\n"
            )
            out.flush()

        _accumulate_sample_summary(summary, result)

    # --- Aggregate summary --------------------------------------------
    if output_dir is None:
        _write_sample_summary(out, summary)
    else:
        # Human-readable summary file with linked ID lists per bucket.
        summary_md_path = output_dir / "summary.md"
        try:
            with summary_md_path.open("w", encoding="utf-8") as f:
                _write_sample_summary_markdown(
                    f, summary, options=options,
                    output_dir=output_dir,
                )
        except OSError as exc:
            summary.failed_to_write_report = True
            err.write(
                f"error: failed to write summary markdown "
                f"{summary_md_path}: {exc}\n"
            )

        # Machine-readable mirror.
        summary_json_path = output_dir / "summary.json"
        try:
            with summary_json_path.open("w", encoding="utf-8") as f:
                json.dump(
                    _serialize_sample_summary(summary, options=options),
                    f, indent=2, default=_default_json_serializer,
                )
                f.write("\n")
        except OSError as exc:
            summary.failed_to_write_report = True
            err.write(
                f"error: failed to write summary JSON "
                f"{summary_json_path}: {exc}\n"
            )

        # Headline rates on stdout so an operator running the CLI
        # without redirecting still sees the bottom line.
        _write_sample_headline(out, summary, output_dir=output_dir)

    return summary


def _accumulate_sample_summary(summary: SampleSummary, result: RunResult) -> None:
    """Update ``summary`` in place with the result's contribution.

    Tracks the per-verdict and per-execution-status counts, the
    multiset-match count, and the two disagreement cells. The
    disagreement cells are the most useful diagnostic — see
    :class:`SampleSummary` for the interpretation.
    """

    summary.results.append(result)
    summary.verdict_counts[result.reported_verdict] = (
        summary.verdict_counts.get(result.reported_verdict, 0) + 1
    )
    if result.execution is not None:
        summary.execution_counts[result.execution.status] = (
            summary.execution_counts.get(result.execution.status, 0) + 1
        )
        if result.execution.multiset_match:
            summary.multiset_match_count += 1
        # Disagreement cells: only meaningful when both signals
        # produced a real verdict. Skip planner_failed / converter_*
        # rows where the cvc5 verdict didn't run, and skip exec
        # status ``skipped`` for the same reason.
        is_logical_yes = (
            result.underlying_verdict == Verdict.equivalent
        )
        is_logical_no = (
            result.underlying_verdict == Verdict.not_equivalent
        )
        is_exec_yes = result.execution.status == ExecutionStatus.match
        is_exec_no = result.execution.status == ExecutionStatus.mismatch
        if is_logical_yes and is_exec_no:
            summary.logical_yes_exec_no += 1
        elif is_logical_no and is_exec_yes:
            summary.logical_no_exec_yes += 1
    else:
        # Execution check disabled: every row falls into the "skipped"
        # bucket so the totals still equal sampled_count.
        summary.execution_counts[ExecutionStatus.skipped] = (
            summary.execution_counts.get(ExecutionStatus.skipped, 0) + 1
        )


def _write_sample_summary(out: TextIO, summary: SampleSummary) -> None:
    """Emit the human-readable aggregate summary at the end of a sample run.

    The format intentionally mirrors what an operator scrolling the
    log wants to see in one screen: the headline rates first
    (logical-equivalent and exec-match percentages), then the
    disagreement cells, then the verdict and execution-status
    breakdowns.
    """

    out.write("\n---\n# Sample Summary\n\n")

    n = max(1, summary.sampled_count)  # avoid div-by-zero for empty sample
    eq_count = summary.verdict_counts.get(Verdict.equivalent, 0)
    exec_match = summary.execution_counts.get(ExecutionStatus.match, 0)

    out.write(
        f"- **Sampled:** {summary.sampled_count} "
        f"(seed={summary.seed}, requested={summary.requested_count})\n"
    )
    out.write(
        f"- **Logically equivalent (cvc5):** "
        f"{eq_count} / {summary.sampled_count} "
        f"({_pct(eq_count, n)})\n"
    )
    out.write(
        f"- **Execution-equivalent (set match):** "
        f"{exec_match} / {summary.sampled_count} "
        f"({_pct(exec_match, n)})\n"
    )
    out.write(
        f"- **Multiset match (strict superset of set match):** "
        f"{summary.multiset_match_count} / {summary.sampled_count} "
        f"({_pct(summary.multiset_match_count, n)})\n"
    )
    out.write(
        f"- **Disagreement (logical=yes, exec=no):** "
        f"{summary.logical_yes_exec_no}\n"
    )
    out.write(
        f"- **Disagreement (logical=no, exec=yes):** "
        f"{summary.logical_no_exec_yes}\n"
    )

    out.write("\n## Verdict breakdown\n\n")
    for verdict in Verdict:
        c = summary.verdict_counts.get(verdict, 0)
        if c:
            out.write(f"- `{verdict.value}`: {c} ({_pct(c, n)})\n")

    out.write("\n## Execution breakdown\n\n")
    for status in ExecutionStatus:
        c = summary.execution_counts.get(status, 0)
        if c:
            out.write(f"- `{status.value}`: {c} ({_pct(c, n)})\n")

    out.write("\n")
    out.flush()


def _pct(numerator: int, denominator: int) -> str:
    """Render ``numerator / denominator`` as a one-decimal percentage."""

    if denominator <= 0:
        return "0.0%"
    return f"{(100.0 * numerator / denominator):.1f}%"


def _write_sample_headline(
    out: TextIO, summary: SampleSummary, *, output_dir: Path
) -> None:
    """Print a short rate summary on stdout when running in directory mode.

    The full summary lives in ``{output_dir}/summary.md``; this is the
    "if you only see one line, see this" version that appears at the
    end of the per-case progress stream so an operator can grok the
    bottom line without opening another file.
    """

    n = max(1, summary.sampled_count)
    eq = summary.verdict_counts.get(Verdict.equivalent, 0)
    em = summary.execution_counts.get(ExecutionStatus.match, 0)
    out.write(
        f"\nsample summary: "
        f"{summary.sampled_count} sampled, "
        f"logical-equivalent={eq} ({_pct(eq, n)}), "
        f"exec-match={em} ({_pct(em, n)}), "
        f"logical-yes/exec-no={summary.logical_yes_exec_no}, "
        f"logical-no/exec-yes={summary.logical_no_exec_yes}\n"
    )
    out.write(f"summary written to {output_dir / 'summary.md'}\n")
    out.flush()


def _write_sample_summary_markdown(
    out: TextIO,
    summary: SampleSummary,
    *,
    options: RunOptions,
    output_dir: Path,
) -> None:
    """Write the human-readable summary file.

    The summary lists, for every non-empty bucket, the actual
    Test_Case_IDs that fell into it, with each ID linked to its
    transcript file (``./{id}.md``, relative-link inside
    ``output_dir``). This is the "list the IDs" requirement: when
    skimming "12 cases were logically equivalent", the operator can
    click straight through to any one of them.

    The summary structure:

    1. Run metadata (seed, sampled / requested counts, split).
    2. Headline rates (logical-equivalent, exec-match, multiset).
    3. Disagreement cells with linked ID lists.
    4. Per-verdict breakdown with linked ID lists.
    5. Per-execution-status breakdown with linked ID lists.
    """

    n = max(1, summary.sampled_count)

    out.write("# BIRD Sample Summary\n\n")
    out.write(f"- **Split:** `{options.split}`\n")
    out.write(
        f"- **Sampled:** {summary.sampled_count} "
        f"(requested {summary.requested_count}, seed={summary.seed})\n"
    )
    out.write(f"- **Output directory:** `{output_dir}`\n\n")

    eq_count = summary.verdict_counts.get(Verdict.equivalent, 0)
    exec_match = summary.execution_counts.get(ExecutionStatus.match, 0)
    out.write("## Headline rates\n\n")
    out.write(
        f"- **Logically equivalent (cvc5):** "
        f"{eq_count} / {summary.sampled_count} "
        f"({_pct(eq_count, n)})\n"
    )
    out.write(
        f"- **Execution-equivalent (set match):** "
        f"{exec_match} / {summary.sampled_count} "
        f"({_pct(exec_match, n)})\n"
    )
    out.write(
        f"- **Multiset match (strict superset of set match):** "
        f"{summary.multiset_match_count} / {summary.sampled_count} "
        f"({_pct(summary.multiset_match_count, n)})\n\n"
    )

    # --- Disagreement cells: linked ID lists ---------------------------
    out.write("## Disagreement cells\n\n")
    yes_no = _ids_in_disagreement(
        summary, logical=Verdict.equivalent,
        exec_status=ExecutionStatus.mismatch,
    )
    no_yes = _ids_in_disagreement(
        summary, logical=Verdict.not_equivalent,
        exec_status=ExecutionStatus.match,
    )
    out.write(
        f"- **logical=yes, exec=no:** {len(yes_no)}"
    )
    if yes_no:
        out.write(": " + _format_id_links(yes_no))
    out.write("\n")
    out.write(
        f"- **logical=no, exec=yes:** {len(no_yes)}"
    )
    if no_yes:
        out.write(": " + _format_id_links(no_yes))
    out.write("\n\n")

    # --- Verdict breakdown ---------------------------------------------
    out.write("## Verdict breakdown\n\n")
    for verdict in Verdict:
        ids = _ids_for_verdict(summary, verdict)
        c = len(ids)
        if c:
            out.write(
                f"### `{verdict.value}` — {c} ({_pct(c, n)})\n\n"
            )
            out.write(_format_id_links(ids) + "\n\n")

    # --- Execution-status breakdown ------------------------------------
    out.write("## Execution-status breakdown\n\n")
    for status in ExecutionStatus:
        ids = _ids_for_exec_status(summary, status)
        c = len(ids)
        if c:
            out.write(
                f"### `{status.value}` — {c} ({_pct(c, n)})\n\n"
            )
            out.write(_format_id_links(ids) + "\n\n")

    out.flush()


def _ids_for_verdict(summary: SampleSummary, verdict: Verdict) -> list[str]:
    """Return the Test_Case_IDs whose ``reported_verdict`` is ``verdict``,
    in sample order.
    """

    return [
        r.test_case_id
        for r in summary.results
        if r.reported_verdict == verdict
    ]


def _ids_for_exec_status(
    summary: SampleSummary, status: ExecutionStatus
) -> list[str]:
    """Return the Test_Case_IDs whose execution status matches, in sample order.

    Cases without an ``execution`` block (the framework was run with
    ``execution_check=False``) all fall under ``ExecutionStatus.skipped``
    so the totals still equal ``sampled_count``.
    """

    out: list[str] = []
    for r in summary.results:
        if r.execution is None:
            if status == ExecutionStatus.skipped:
                out.append(r.test_case_id)
            continue
        if r.execution.status == status:
            out.append(r.test_case_id)
    return out


def _ids_in_disagreement(
    summary: SampleSummary,
    *,
    logical: Verdict,
    exec_status: ExecutionStatus,
) -> list[str]:
    """Return the Test_Case_IDs in a specific (logical, exec) cell."""

    out: list[str] = []
    for r in summary.results:
        if r.execution is None:
            continue
        if r.underlying_verdict == logical and r.execution.status == exec_status:
            out.append(r.test_case_id)
    return out


def _format_id_links(ids: list[str]) -> str:
    """Render ``ids`` as a comma-separated list of ``[id](./id.md)`` links.

    The operator opens the summary in a markdown viewer (or just a
    text editor) and can click straight through to a case file.
    """

    return ", ".join(f"[{tid}](./{tid}.md)" for tid in ids)


def _serialize_sample_summary(
    summary: SampleSummary, *, options: RunOptions
) -> dict[str, Any]:
    """Convert ``summary`` into the JSON shape written to ``summary.json``.

    Contains the headline numbers, the disagreement cells with ID
    lists, and a per-bucket ID list under each verdict and execution
    status. Unlike the markdown summary, this is meant for scripts /
    dashboards — the IDs are bare strings, not linked.
    """

    n = max(1, summary.sampled_count)
    eq = summary.verdict_counts.get(Verdict.equivalent, 0)
    em = summary.execution_counts.get(ExecutionStatus.match, 0)

    verdict_breakdown: dict[str, dict[str, Any]] = {}
    for verdict in Verdict:
        ids = _ids_for_verdict(summary, verdict)
        verdict_breakdown[verdict.value] = {
            "count": len(ids),
            "ids": ids,
        }

    exec_breakdown: dict[str, dict[str, Any]] = {}
    for status in ExecutionStatus:
        ids = _ids_for_exec_status(summary, status)
        exec_breakdown[status.value] = {
            "count": len(ids),
            "ids": ids,
        }

    yes_no = _ids_in_disagreement(
        summary, logical=Verdict.equivalent,
        exec_status=ExecutionStatus.mismatch,
    )
    no_yes = _ids_in_disagreement(
        summary, logical=Verdict.not_equivalent,
        exec_status=ExecutionStatus.match,
    )

    return {
        "split": options.split,
        "seed": summary.seed,
        "requested_count": summary.requested_count,
        "sampled_count": summary.sampled_count,
        "rates": {
            "logical_equivalent": {
                "count": eq,
                "pct": _pct(eq, n),
            },
            "execution_match": {
                "count": em,
                "pct": _pct(em, n),
            },
            "multiset_match": {
                "count": summary.multiset_match_count,
                "pct": _pct(summary.multiset_match_count, n),
            },
        },
        "disagreements": {
            "logical_yes_exec_no": {
                "count": len(yes_no),
                "ids": yes_no,
            },
            "logical_no_exec_yes": {
                "count": len(no_yes),
                "ids": no_yes,
            },
        },
        "verdict_breakdown": verdict_breakdown,
        "execution_breakdown": exec_breakdown,
        # Always include the ordered list of sampled IDs so a re-run
        # with the same seed can be cross-checked even if the loader's
        # ordering ever changes.
        "sampled_ids": [r.test_case_id for r in summary.results],
    }
