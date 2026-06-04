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
below), the public DRC type ``DRCExpression`` and the public helper
``query_inner_drc`` from ``types.drc``. The :class:`EquivalenceResult`
union members are *not* imported: the runner dispatches on the public
``result.status`` literal field instead, so the result-class identities
stay an implementation detail of the equivalence checker.
"""

from __future__ import annotations

import asyncio
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
from text_to_sql_planner.main import (
    TextToSQLFailure,
    TextToSQLSuccess,
    run as _default_planner_run,
)
from text_to_sql_planner.types.drc import DRCExpression, query_inner_drc

from bird_benchmark.evidence import forward_call
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
    RunOptions,
    RunResult,
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


async def run_one(
    test_case: TestCase,
    options: RunOptions,
    expected_fail: set[str],
    *,
    planner_callable: Callable[..., Awaitable[Any]] = _default_planner_run,
    convert_sql_callable: Callable[[str, str], Any] = _default_convert_sql,
    check_equivalence_callable: Callable[..., Awaitable[Any]] = _default_check_equivalence,
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
        return _apply_expected_fail(
            RunResult(
                **base,
                underlying_verdict=Verdict.planner_failed,
                reported_verdict=Verdict.planner_failed,
                reason=_truncate_reason("planner_timeout"),
                error_code="planner_timeout",
            ),
            expected_fail,
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
        return _apply_expected_fail(
            RunResult(
                **base,
                underlying_verdict=Verdict.planner_failed,
                reported_verdict=Verdict.planner_failed,
                reason=_truncate_reason(f"{code}: {message}"),
                error_code=code,
            ),
            expected_fail,
        )

    if isinstance(planner_result, TextToSQLFailure):
        # Req 3.3 / 5.7: the planner returned a structured failure.
        # Surface both the message and the code in ``reason`` so
        # operators do not have to cross-reference the JSON report's
        # ``error_code`` field to make sense of the markdown.
        message = _safe_message(planner_result.error)
        code = _safe_code(planner_result.code)
        return _apply_expected_fail(
            RunResult(
                **base,
                underlying_verdict=Verdict.planner_failed,
                reported_verdict=Verdict.planner_failed,
                reason=_truncate_reason(f"{code}: {message}"),
                error_code=code,
            ),
            expected_fail,
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
        return _apply_expected_fail(
            RunResult(
                **base,
                underlying_verdict=Verdict.planner_failed,
                reported_verdict=Verdict.planner_failed,
                reason=_truncate_reason(f"{code}: {message}"),
                error_code=code,
            ),
            expected_fail,
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
        return _apply_expected_fail(
            RunResult(
                **base,
                underlying_verdict=verdict,
                reported_verdict=verdict,
                reason=_truncate_reason(reason_text),
            ),
            expected_fail,
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
        return _apply_expected_fail(
            RunResult(
                **base,
                underlying_verdict=Verdict.gold_conversion_failure,
                reported_verdict=Verdict.gold_conversion_failure,
                reason=_truncate_reason(_safe_message(str(exc))),
            ),
            expected_fail,
        )

    # --- Step 4: Equivalence check -----------------------------------
    config = EquivalenceCheckerConfig(
        timeout_seconds=options.cvc5_timeout_seconds,
        cvc5_path=options.cvc5_path,
    )
    eq_result = await check_equivalence_callable(
        generated_drc, gold_drc, config
    )

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

    return _apply_expected_fail(
        RunResult(
            **base,
            underlying_verdict=verdict,
            reported_verdict=verdict,
            smt_script=smt_script,
            reason=reason_text,
        ),
        expected_fail,
    )


__all__ = ["run_one", "run_single", "run_suite", "SingleSelectorError"]


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

    result = await run_one(
        test_case,
        options,
        expected_fail,
        planner_callable=planner_callable,
        convert_sql_callable=convert_sql_callable,
        check_equivalence_callable=check_equivalence_callable,
    )

    out = stdout if stdout is not None else sys.stdout
    payload = _serialize_run_result(result)
    out.write(json.dumps(payload, default=_default_json_serializer))
    out.write("\n")
    out.flush()

    return result


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
