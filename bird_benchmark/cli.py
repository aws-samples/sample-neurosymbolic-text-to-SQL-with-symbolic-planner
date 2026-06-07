"""Command-line entry point for the BIRD benchmark framework.

This module wires the public ``bird-benchmark`` CLI specified in the
design's CLI section. It uses ``argparse`` with two subcommands —
``single`` and ``suite`` — and dispatches to
:func:`bird_benchmark.runner.run_single` or
:func:`bird_benchmark.runner.run_suite` accordingly. The argument layout
mirrors the design verbatim:

* ``single --split --bird-root (--id | --question) [common opts]``
* ``suite  --split --bird-root --manifest [--resume] [common opts]``

Common options live on both subcommands: ``--cvc5-path``,
``--cvc5-timeout``, ``--per-test-timeout``, and ``--expected-fail``.
``--report-json`` and ``--report-md`` are suite-only because single-mode
does not produce reports (it prints a single :class:`RunResult` to
stdout instead, per Req 7.3).

Validation
----------

Per Req 6.2 the CLI must reject ``--cvc5-timeout`` / ``--per-test-timeout``
values that are not integers in ``[1, 3600]`` *before the suite starts*,
with an error message that names the offending value. Both validations
live in :func:`_timeout_int`, an :mod:`argparse` ``type`` callback that
raises :class:`argparse.ArgumentTypeError` so argparse's standard
non-zero-exit usage path runs.

Single-mode selector validation (``--id`` xor ``--question``) is left to
the runner's :func:`bird_benchmark.runner._validate_selector` helper. Its
three error messages already match Req 7.4 / 7.5 / 7.6, and going via
:class:`SingleSelectorError` keeps the CLI thin: it catches the exception
and exits non-zero.

Exit codes
----------

* ``0`` on success.
* ``1`` for single-mode selector failures (Req 7.4 / 7.5 / 7.6) via
  :class:`SingleSelectorError`.
* ``2`` for missing or malformed configuration:

  - invalid ``--cvc5-timeout`` / ``--per-test-timeout`` values
    (Req 6.2) — argparse's own ``ArgumentTypeError`` path already
    exits with code ``2``.
  - missing BIRD JSON or per-record SQLite (Req 1.5 / 1.6) via
    :class:`BirdLoadError`.
  - missing or unreadable Expected_Fail_List (Req 12.2) via
    :class:`ExpectedFailLoadError`.
  - malformed manifest under ``--resume`` (Req 8.5) via
    :class:`ManifestParseError`.

* ``3`` when one or both reports could not be written
  (``SuiteSummary.failed_to_write_report``, Req 9.6).

The :func:`main` function takes an optional ``argv`` parameter so unit
tests can drive the CLI without touching :data:`sys.argv` and returns
the integer exit code so tests can assert on it directly. The
``__main__.py`` shim wraps the call in :func:`sys.exit` for the actual
``python -m bird_benchmark`` / ``bird-benchmark`` entry points.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence, TextIO

from bird_benchmark.expected_fail import ExpectedFailLoadError
from bird_benchmark.installer import (
    InstallError,
    InstallReport,
    install_split,
)
from bird_benchmark.loader import BirdLoader, BirdLoaderConfig, BirdLoadError
from bird_benchmark.manifest import ManifestParseError
from bird_benchmark.runner import (
    SingleSelectorError,
    run_sample,
    run_single,
    run_suite,
)
from bird_benchmark.types import RunOptions, SingleSelector, SkippedTestCase


# Inclusive bounds for the two operator-supplied timeouts (Req 6.1 / 6.2 /
# 6.6). The default-value side of the contract lives in :mod:`types` on
# :class:`RunOptions`; the validation side lives here so a bad value is
# rejected *before* any I/O happens — argparse parsing runs before the
# loader is touched.
_TIMEOUT_MIN_SECONDS = 1
_TIMEOUT_MAX_SECONDS = 3600

# Exit codes (see module docstring). Centralised here so tests and the
# ``__main__`` shim can both import them and the task contract stays in
# one place.
EXIT_OK = 0
EXIT_SELECTOR_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_REPORT_WRITE_ERROR = 3

# Default report paths used when the operator does not pass
# ``--report-json`` / ``--report-md`` on the ``suite`` subcommand. These
# match the values called out in the design's CLI section.
_DEFAULT_REPORT_JSON_PATH = Path("bird-report.json")
_DEFAULT_REPORT_MD_PATH = Path("bird-report.md")


def _timeout_int(value: str) -> int:
    """argparse ``type`` callback for the two timeout flags (Req 6.2).

    Accepts only integer strings in the inclusive range
    ``[_TIMEOUT_MIN_SECONDS, _TIMEOUT_MAX_SECONDS]``. Anything else
    raises :class:`argparse.ArgumentTypeError`, which argparse converts
    into a non-zero-exit usage error whose message names the offending
    value. The error message includes the original spelling (via
    ``repr``) so an operator can spot whitespace or unit-suffix typos.
    """

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"timeout must be an integer in "
            f"[{_TIMEOUT_MIN_SECONDS}, {_TIMEOUT_MAX_SECONDS}]; got {value!r}"
        )
    if not (_TIMEOUT_MIN_SECONDS <= parsed <= _TIMEOUT_MAX_SECONDS):
        raise argparse.ArgumentTypeError(
            f"timeout must be an integer in "
            f"[{_TIMEOUT_MIN_SECONDS}, {_TIMEOUT_MAX_SECONDS}]; got {parsed}"
        )
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with ``single`` / ``suite``.

    A ``common`` parent parser holds the four flags shared between the
    two subcommands (``--split``, ``--bird-root``, ``--cvc5-path``,
    ``--cvc5-timeout``, ``--per-test-timeout``, ``--expected-fail``) so
    the help text stays in one place. The subcommand-specific flags are
    added on each child parser directly.
    """

    parser = argparse.ArgumentParser(
        prog="bird-benchmark",
        description=(
            "Run the BIRD benchmark against the text_to_sql_planner. "
            "Use ``single`` to evaluate one Test_Case for fast iteration "
            "or ``suite`` to evaluate an entire BIRD split with resume."
        ),
    )

    # Required subcommand selector. ``required=True`` makes argparse exit
    # non-zero with a usage error when the operator forgets the mode,
    # which is the behaviour Req 10.2 demands ("exactly one mode is
    # active per invocation").
    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        metavar="MODE",
        title="modes",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--split",
        required=True,
        metavar="SPLIT",
        help="BIRD split name as it appears on disk (e.g. dev, test).",
    )
    common.add_argument(
        "--bird-root",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Directory containing the BIRD download. The split JSON is "
            "expected at {bird-root}/{split}/{split}.json."
        ),
    )
    common.add_argument(
        "--cvc5-path",
        default="cvc5",
        metavar="PATH",
        help="cvc5 executable path or command name (default: cvc5).",
    )
    common.add_argument(
        "--cvc5-timeout",
        type=_timeout_int,
        default=30,
        metavar="SECONDS",
        help=(
            "Per-query cvc5 timeout, integer seconds in "
            f"[{_TIMEOUT_MIN_SECONDS}, {_TIMEOUT_MAX_SECONDS}] (default: 30)."
        ),
    )
    common.add_argument(
        "--per-test-timeout",
        type=_timeout_int,
        default=60,
        metavar="SECONDS",
        help=(
            "Per-Test_Case wall-clock budget on the planner, integer "
            f"seconds in [{_TIMEOUT_MIN_SECONDS}, {_TIMEOUT_MAX_SECONDS}] "
            "(default: 60)."
        ),
    )
    common.add_argument(
        "--expected-fail",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to an Expected_Fail_List file (one Test_Case_ID per "
            "line). Listed Test_Cases whose underlying verdict is not "
            "equivalent are reported as ``expected_fail``. Defaults to "
            "no list."
        ),
    )
    common.add_argument(
        "--no-execution-check",
        dest="execution_check",
        action="store_false",
        default=True,
        help=(
            "Skip running the generated and gold SQL against the BIRD "
            "SQLite database. By default the framework records both "
            "the cvc5 verdict and the executed-row-set verdict so the "
            "report can show their (dis)agreement."
        ),
    )
    common.add_argument(
        "--exec-timeout",
        type=_timeout_int,
        default=30,
        metavar="SECONDS",
        help=(
            "Per-query timeout for the execution check, integer "
            f"seconds in [{_TIMEOUT_MIN_SECONDS}, {_TIMEOUT_MAX_SECONDS}] "
            "(default: 30). Independent of --cvc5-timeout — the two "
            "checks are unrelated work."
        ),
    )

    # --- single subcommand -------------------------------------------
    single = subparsers.add_parser(
        "single",
        parents=[common],
        help="Run exactly one Test_Case selected by ID or question text.",
        description=(
            "Run a single BIRD Test_Case end-to-end and print its full "
            "Run_Result as JSON to stdout. Exactly one of --id or "
            "--question must be supplied (Req 7.6)."
        ),
    )
    # The exactly-one rule is enforced by the runner's
    # ``_validate_selector`` helper rather than by an argparse mutually
    # exclusive group: argparse's group error message ("argument --id:
    # not allowed with argument --question") does not match the
    # framework's standard SingleSelectorError wording, and a required
    # group rejects "neither" with a generic argparse usage error
    # instead of the framework's "got neither" message. Pushing the
    # check to the runner keeps a single, consistent error contract for
    # both library and CLI callers.
    single.add_argument(
        "--id",
        dest="test_case_id",
        default=None,
        metavar="TEST_CASE_ID",
        help="Select the Test_Case whose Test_Case_ID equals this value.",
    )
    single.add_argument(
        "--question",
        dest="question_text",
        default=None,
        metavar="TEXT",
        help=(
            "Select the Test_Case whose question text equals this "
            "value (exact, case-sensitive)."
        ),
    )

    # --- suite subcommand --------------------------------------------
    suite = subparsers.add_parser(
        "suite",
        parents=[common],
        help="Run an entire BIRD split with append-only manifest and resume.",
        description=(
            "Process every Test_Case in the requested split. Each "
            "Run_Result is appended to the JSONL manifest before the "
            "next Test_Case begins so a SIGKILL can be recovered with "
            "--resume."
        ),
    )
    suite.add_argument(
        "--manifest",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to the append-only Run_Manifest JSONL file.",
    )
    suite.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Read the manifest at --manifest, skip Test_Cases whose IDs "
            "already appear, and continue from where the previous run "
            "stopped. Starts a fresh suite when the manifest does not "
            "yet exist (Req 8.7)."
        ),
    )
    suite.add_argument(
        "--report-json",
        type=Path,
        default=_DEFAULT_REPORT_JSON_PATH,
        metavar="PATH",
        help=(
            "Path to the JSON report written at suite end "
            f"(default: {_DEFAULT_REPORT_JSON_PATH})."
        ),
    )
    suite.add_argument(
        "--report-md",
        type=Path,
        default=_DEFAULT_REPORT_MD_PATH,
        metavar="PATH",
        help=(
            "Path to the markdown report written at suite end "
            f"(default: {_DEFAULT_REPORT_MD_PATH})."
        ),
    )

    # --- sample subcommand -------------------------------------------
    sample = subparsers.add_parser(
        "sample",
        parents=[common],
        help="Run a deterministic random sample of N Test_Cases from a split.",
        description=(
            "Draw N Test_Cases from the requested split using a "
            "user-supplied seed and run each one through the same "
            "pipeline ``single`` uses. After every Test_Case has run, "
            "an aggregate summary reports both the cvc5 logical-"
            "equivalence rate and the SQLite execution-equivalence "
            "rate, plus the cells where the two signals disagree. "
            "Useful for diagnostic runs where the full suite is too "
            "slow but a hand-picked single case is too narrow."
        ),
    )
    sample.add_argument(
        "--count",
        required=True,
        type=int,
        metavar="N",
        help=(
            "Number of Test_Cases to sample. When the split has "
            "fewer eligible Test_Cases than N the sampler runs "
            "every eligible record once."
        ),
    )
    sample.add_argument(
        "--seed",
        required=True,
        type=int,
        metavar="N",
        help=(
            "PRNG seed used to draw the sample. Required so two "
            "runs with the same split + count + seed produce the "
            "same selection — without that you cannot tell apart "
            "\"my code got better\" from \"I sampled different "
            "cases this time\"."
        ),
    )
    sample.add_argument(
        "--report-json",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional path to write a JSON report of the sample. "
            "Defaults to no report — sampling is meant for ad-hoc "
            "diagnostic runs and the aggregate summary printed to "
            "stdout is usually enough."
        ),
    )
    sample.add_argument(
        "--report-md",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional path to write a markdown report of the sample. "
            "Defaults to no report."
        ),
    )

    # --- install subcommand ------------------------------------------
    # The install subcommand intentionally does NOT inherit ``common``:
    # it doesn't run cvc5, doesn't load the planner, and has no use for
    # ``--cvc5-path`` / ``--cvc5-timeout`` / ``--per-test-timeout`` /
    # ``--expected-fail``. Reusing ``common`` would also force operators
    # to pass ``--split`` twice (once via common, once on a child).
    install = subparsers.add_parser(
        "install",
        help="Download and install a BIRD split into a local directory.",
        description=(
            "Download the official BIRD ZIP for the requested split, "
            "extract it, normalise the on-disk layout to what the "
            "loader expects, and verify the result by streaming every "
            "record once. After this completes, the same --bird-root "
            "and --split can be passed to ``single`` / ``suite``."
        ),
    )
    install.add_argument(
        "--bird-root",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Directory that will hold the installed split. Created if "
            "missing. After this call, {bird-root}/{split}/{split}.json "
            "exists."
        ),
    )
    install.add_argument(
        "--split",
        required=True,
        metavar="SPLIT",
        help=(
            "BIRD split to install (e.g. dev, train). The split name "
            "becomes the directory name under --bird-root."
        ),
    )
    install.add_argument(
        "--url",
        default=None,
        metavar="URL",
        help=(
            "Override the download URL. Defaults to the official BIRD "
            "endpoint for the chosen split. Only https:// and file:// "
            "schemes are accepted."
        ),
    )
    install.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download and re-install even when {bird-root}/{split} "
            "already contains a valid layout. Without --force the "
            "command short-circuits to verification when the install "
            "is already in place."
        ),
    )
    install.add_argument(
        "--keep-archive",
        action="store_true",
        help=(
            "Keep the downloaded ZIP at {bird-root}/_downloads/{split}.zip "
            "after extraction so a future --force can re-extract without "
            "re-downloading. Default: archive is removed once the "
            "layout is in place."
        ),
    )

    # --- list subcommand ---------------------------------------------
    # ``list`` doesn't run the planner or cvc5 either, so it shares
    # the install subcommand's pattern of declaring its own
    # ``--bird-root`` / ``--split`` rather than inheriting ``common``.
    # The job is purely "what IDs can I pass to ``single --id``?", so
    # the only real flags are presentational filters.
    list_cmd = subparsers.add_parser(
        "list",
        help="List the Test_Case IDs available in an installed BIRD split.",
        description=(
            "Print one line per Test_Case in the requested split. The "
            "default format is ``{test_case_id}\\t{db_id}\\t{question}`` "
            "so the output can be piped through grep/awk/fzf to pick an "
            "--id for ``bird-benchmark single``. Records the loader had "
            "to skip (missing field, missing SQLite) are reported on "
            "stderr but don't show up in the listing."
        ),
    )
    list_cmd.add_argument(
        "--bird-root",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Directory containing the installed BIRD split (the same "
            "value passed to ``install --bird-root``)."
        ),
    )
    list_cmd.add_argument(
        "--split",
        required=True,
        metavar="SPLIT",
        help="BIRD split name to list (e.g. dev, train).",
    )
    list_cmd.add_argument(
        "--db",
        default=None,
        metavar="DB_ID",
        help=(
            "Optional filter: only list Test_Cases whose ``db_id`` "
            "exactly matches this value."
        ),
    )
    list_cmd.add_argument(
        "--contains",
        default=None,
        metavar="SUBSTRING",
        help=(
            "Optional filter: only list Test_Cases whose question "
            "contains this substring (case-insensitive)."
        ),
    )
    list_cmd.add_argument(
        "--ids-only",
        action="store_true",
        help=(
            "Print only the Test_Case_ID per line, with no db_id or "
            "question. Useful for piping into ``xargs`` or seeding an "
            "Expected_Fail_List."
        ),
    )
    list_cmd.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop after listing N Test_Cases. Pairs well with "
            "``--contains`` for quick previews."
        ),
    )

    return parser


def _options_from_args(args: argparse.Namespace) -> RunOptions:
    """Translate parsed argparse ``Namespace`` into a :class:`RunOptions`.

    Suite-only fields (``manifest_path``, ``resume``,
    ``report_json_path``, ``report_md_path``) are only populated when the
    suite subcommand was selected; in single mode they stay ``None`` /
    ``False`` so the runner's library-mode safety check on
    ``manifest_path`` only fires for genuine misuse.
    """

    options = RunOptions(
        bird_root=args.bird_root,
        split=args.split,
        cvc5_timeout_seconds=args.cvc5_timeout,
        per_test_timeout_seconds=args.per_test_timeout,
        expected_fail_path=args.expected_fail,
        cvc5_path=args.cvc5_path,
        execution_check=args.execution_check,
        execution_timeout_seconds=args.exec_timeout,
    )
    if args.mode == "suite":
        options.manifest_path = args.manifest
        options.resume = args.resume
        options.report_json_path = args.report_json
        options.report_md_path = args.report_md
    elif args.mode == "sample":
        # Sample mode reuses the same RunOptions report fields but
        # defaults to ``None`` (no report) — the operator opts in.
        options.report_json_path = args.report_json
        options.report_md_path = args.report_md
    return options


def _print_error(message: str, stderr: TextIO) -> None:
    """Emit a single error line to ``stderr`` with a consistent prefix.

    The ``error: `` prefix matches the framework's existing convention
    (the suite driver writes ``error: failed to write …`` for report
    failures and ``warning: …`` for stale-entry warnings) so report
    consumers can grep for ``^error:`` regardless of which layer raised.
    """

    stderr.write(f"error: {message}\n")
    stderr.flush()


def _run_single(args: argparse.Namespace, stderr: TextIO) -> int:
    """Dispatch the ``single`` subcommand. Returns the integer exit code."""

    options = _options_from_args(args)
    selector = SingleSelector(
        test_case_id=args.test_case_id,
        question_text=args.question_text,
    )
    try:
        asyncio.run(run_single(selector, options))
    except SingleSelectorError as exc:
        # Covers Req 7.4 / 7.5 / 7.6: zero / many / neither / both
        # selector cases. The exception's message already names the
        # selector and the split per the runner's contract.
        _print_error(exc.message, stderr)
        return EXIT_SELECTOR_ERROR
    except ExpectedFailLoadError as exc:
        # Req 12.2: missing or unreadable Expected_Fail_List.
        _print_error(str(exc), stderr)
        return EXIT_CONFIG_ERROR
    except BirdLoadError as exc:
        # Req 1.5 / 1.6: missing BIRD JSON or unparseable JSON.
        _print_error(str(exc), stderr)
        return EXIT_CONFIG_ERROR
    return EXIT_OK


def _run_suite(args: argparse.Namespace, stderr: TextIO) -> int:
    """Dispatch the ``suite`` subcommand. Returns the integer exit code."""

    options = _options_from_args(args)
    try:
        summary = asyncio.run(run_suite(options))
    except ManifestParseError as exc:
        # Req 8.5: malformed manifest under --resume. The exception's
        # ``__str__`` already names the path and the parse error.
        _print_error(str(exc), stderr)
        return EXIT_CONFIG_ERROR
    except ExpectedFailLoadError as exc:
        # Req 12.2.
        _print_error(str(exc), stderr)
        return EXIT_CONFIG_ERROR
    except BirdLoadError as exc:
        # Req 1.5 / 1.6.
        _print_error(str(exc), stderr)
        return EXIT_CONFIG_ERROR
    if summary.failed_to_write_report:
        # Req 9.6: the suite driver already emitted a per-report
        # ``error: failed to write …`` line to stderr; the CLI's job is
        # only to surface the failure as a non-zero exit so wrapper
        # scripts can distinguish a successful report write from a
        # silently-broken one.
        return EXIT_REPORT_WRITE_ERROR
    return EXIT_OK


def _run_sample(
    args: argparse.Namespace, stdout: TextIO, stderr: TextIO
) -> int:
    """Dispatch the ``sample`` subcommand. Returns the integer exit code.

    Sample mode shares the same error surface as ``suite`` for
    config / load failures (``EXIT_CONFIG_ERROR``) and report-write
    failures (``EXIT_REPORT_WRITE_ERROR``). The
    :class:`ValueError`-based validation of ``--count`` / ``--seed``
    inside :func:`run_sample` also lands in ``EXIT_CONFIG_ERROR``;
    invalid integers caught by argparse already exit ``2`` via
    argparse's own error path.
    """

    options = _options_from_args(args)
    try:
        summary = asyncio.run(
            run_sample(
                options,
                count=args.count,
                seed=args.seed,
                stdout=stdout,
                stderr=stderr,
            )
        )
    except ValueError as exc:
        _print_error(str(exc), stderr)
        return EXIT_CONFIG_ERROR
    except ExpectedFailLoadError as exc:
        _print_error(str(exc), stderr)
        return EXIT_CONFIG_ERROR
    except BirdLoadError as exc:
        _print_error(str(exc), stderr)
        return EXIT_CONFIG_ERROR
    if summary.failed_to_write_report:
        return EXIT_REPORT_WRITE_ERROR
    return EXIT_OK


def _format_bytes(num_bytes: int) -> str:
    """Render a byte count as a short human-friendly string.

    Used by the ``install`` progress callback so a multi-GB download
    doesn't scroll the terminal with raw byte counts. Stays in binary
    units (MiB, GiB) because that's what most ZIP-distribution servers
    report and what an operator comparing against a Content-Length
    header would expect.
    """

    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ("KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} PiB"


def _make_install_progress(stdout: TextIO):
    """Build a progress callback that reports download progress to ``stdout``.

    Throttling is handled inside the installer (it only invokes the
    callback every ~0.5 s or 16 MiB, whichever happens first) so we
    don't add another layer here. Each tick rewrites the same line via
    ``\\r`` so a long download stays on one row of the terminal; we
    drop a final newline at 100 percent so the next install-stage
    message starts on a fresh line.
    """

    def progress(done: int, total: int) -> None:
        if total > 0:
            percent = (done * 100) // total if total else 0
            line = (
                f"\rdownloading: {_format_bytes(done)} / "
                f"{_format_bytes(total)} ({percent:>3d}%)"
            )
        else:
            line = f"\rdownloading: {_format_bytes(done)}"
        stdout.write(line)
        stdout.flush()
        if total > 0 and done >= total:
            stdout.write("\n")
            stdout.flush()

    return progress


def _run_install(
    args: argparse.Namespace, stdout: TextIO, stderr: TextIO
) -> int:
    """Dispatch the ``install`` subcommand. Returns the integer exit code.

    All :class:`InstallError` failures map to :data:`EXIT_CONFIG_ERROR`
    so the CLI's error contract stays consistent with the rest of the
    framework: any "the install couldn't be set up" condition lands in
    code 2 alongside missing JSON, missing SQLite, and unreadable
    expected-fail files. The exception's ``stage`` is included in the
    error message so an operator can tell whether to retry the
    download, fix permissions, or report a BIRD-side schema change.
    """

    progress = _make_install_progress(stdout)
    try:
        report = install_split(
            bird_root=args.bird_root,
            split=args.split,
            url=args.url,
            force=args.force,
            keep_archive=args.keep_archive,
            progress=progress,
        )
    except InstallError as exc:
        # Make sure we land on a fresh line if the progress callback
        # left the terminal mid-row.
        stdout.write("\n")
        stdout.flush()
        _print_error(f"[install:{exc.stage}] {exc.message}", stderr)
        return EXIT_CONFIG_ERROR

    # Success summary. One line so wrapper scripts can grep for
    # ``installed split=…``; notes are appended afterwards on their own
    # lines so multi-line idiosyncrasies don't break the grep.
    stdout.write(
        f"installed split={report.split} at {report.split_root} "
        f"(databases={report.database_count}, "
        f"test_cases={report.test_case_count}, "
        f"skipped={report.skipped_records}, "
        f"downloaded={_format_bytes(report.download_bytes)})\n"
    )
    for note in report.notes:
        stdout.write(f"note: {note}\n")
    stdout.flush()
    return EXIT_OK


def _run_list(
    args: argparse.Namespace, stdout: TextIO, stderr: TextIO
) -> int:
    """Dispatch the ``list`` subcommand. Returns the integer exit code.

    Streams the BIRD split through :class:`BirdLoader` and prints one
    line per Test_Case to stdout. Loader-level failures (missing JSON,
    unparseable JSON) map to :data:`EXIT_CONFIG_ERROR` for parity with
    ``single`` / ``suite``; per-record skips (missing field, missing
    SQLite) are surfaced on stderr as ``warning:`` lines rather than
    fatal errors so a partially-broken install can still be browsed.
    """

    db_filter: str | None = args.db
    contains_filter: str | None = (
        args.contains.lower() if args.contains else None
    )
    limit: int | None = args.limit
    ids_only: bool = args.ids_only

    if limit is not None and limit < 0:
        _print_error(f"--limit must be non-negative, got {limit}", stderr)
        return EXIT_CONFIG_ERROR

    config = BirdLoaderConfig(bird_root=args.bird_root, split=args.split)
    loader = BirdLoader(config)
    try:
        items = loader.load()
        printed = 0
        skipped = 0
        for item in items:
            if isinstance(item, SkippedTestCase):
                # Skip records get a warning so the operator knows the
                # split has malformed entries, but they don't pollute
                # the ID listing on stdout.
                stderr.write(
                    f"warning: skipping {item.test_case_id}: {item.reason}\n"
                )
                skipped += 1
                continue
            if db_filter is not None and item.db_id != db_filter:
                continue
            if contains_filter is not None and contains_filter not in item.question.lower():
                continue
            if ids_only:
                stdout.write(f"{item.test_case_id}\n")
            else:
                stdout.write(
                    f"{item.test_case_id}\t{item.db_id}\t{item.question}\n"
                )
            printed += 1
            if limit is not None and printed >= limit:
                break
    except BirdLoadError as exc:
        _print_error(str(exc), stderr)
        return EXIT_CONFIG_ERROR

    if skipped:
        stderr.write(
            f"warning: {skipped} record(s) were skipped; "
            f"see ``warning: skipping ...`` lines above\n"
        )
        stderr.flush()
    stdout.flush()
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    stderr: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run the ``bird-benchmark`` CLI and return the integer exit code.

    Parameters
    ----------
    argv:
        Optional argument vector. ``None`` (the default) lets argparse
        read from :data:`sys.argv`; pass a list (e.g. ``["single",
        "--split", "dev", ...]``) to drive the CLI from a unit test.
    stderr:
        Optional stream for error output. Defaults to :data:`sys.stderr`;
        tests can pass an :class:`io.StringIO` to capture without
        monkeypatching.
    stdout:
        Optional stream for normal output. Defaults to
        :data:`sys.stdout`. Today this only matters for the
        ``install`` subcommand, which writes its progress + summary
        here.

    Returns
    -------
    int
        ``0`` on success, non-zero on any error path enumerated in the
        module docstring.
    """

    err = stderr if stderr is not None else sys.stderr
    out = stdout if stdout is not None else sys.stdout
    parser = _build_parser()
    # ``parse_args`` raises ``SystemExit(2)`` on argparse usage errors
    # (invalid timeout, unknown mode, missing required argument); we let
    # that propagate so the existing argparse error reporting does its
    # job. ``main`` is only responsible for the framework-level error
    # paths below.
    args = parser.parse_args(argv)

    if args.mode == "single":
        return _run_single(args, err)
    if args.mode == "suite":
        return _run_suite(args, err)
    if args.mode == "install":
        return _run_install(args, out, err)
    if args.mode == "list":
        return _run_list(args, out, err)
    if args.mode == "sample":
        return _run_sample(args, out, err)

    # ``required=True`` on the subparser group makes this branch
    # unreachable, but defensively returning a non-zero exit keeps the
    # CLI honest if a future refactor ever drops that ``required``.
    _print_error(f"unknown mode {args.mode!r}", err)
    return EXIT_CONFIG_ERROR


__all__ = [
    "main",
    "EXIT_OK",
    "EXIT_SELECTOR_ERROR",
    "EXIT_CONFIG_ERROR",
    "EXIT_REPORT_WRITE_ERROR",
]
