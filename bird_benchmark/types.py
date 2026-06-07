"""Shared types for the BIRD benchmark framework.

This module defines the data classes and enumerations shared by every
component in the ``bird_benchmark`` package. The shapes here come straight
from the ``Data Models`` section of ``design.md``; please update both this
module and the design document together when fields change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


class Verdict(str, Enum):
    """The nine verdict categories produced by the framework.

    Inheriting from ``str`` makes JSON serialization round-trip through
    ``json.dumps`` without a custom encoder: ``Verdict.equivalent ==
    "equivalent"`` is true.
    """

    equivalent = "equivalent"
    not_equivalent = "not_equivalent"
    unknown = "unknown"
    planner_failed = "planner_failed"
    converter_failed = "converter_failed"
    skipped = "skipped"
    expected_fail = "expected_fail"
    timeout = "timeout"
    gold_conversion_failure = "gold_conversion_failure"


class ExecutionStatus(str, Enum):
    """Outcome of running both SQL queries against the BIRD SQLite DB.

    This is a separate signal from :class:`Verdict`: the verdict is a
    proof over all possible database states (cvc5), while the
    execution status is an observation on the one database BIRD
    ships. They tell different things — when they disagree it
    usually means the gold query has a constraint that's vacuous on
    this database (e.g. a JOIN on a foreign key whose values all
    happen to match), and that disagreement is itself the most
    useful diagnostic.
    """

    match = "match"
    """Both queries ran and their result sets agree."""

    mismatch = "mismatch"
    """Both queries ran but their result sets disagree."""

    generated_error = "generated_error"
    """The generated query raised a SQLite error.

    Most common cause: the planner emitted Postgres / standard-SQL
    constructs that SQLite doesn't accept (``DATE 'YYYY-MM-DD'``
    literals, ``INTERVAL`` arithmetic, etc.).
    """

    gold_error = "gold_error"
    """The gold query raised a SQLite error.

    Rare but happens — BIRD has a few records whose gold SQL
    references columns that exist in the schema but not in this
    database snapshot.
    """

    timeout = "timeout"
    """One of the queries did not complete within the timeout budget."""

    db_unavailable = "db_unavailable"
    """The per-database SQLite file could not be opened."""

    skipped = "skipped"
    """Execution check was skipped.

    Set when there is no generated SQL to run (planner failed,
    gold-conversion failed) or when the operator turned exec checks
    off via ``--no-execution-check``.
    """


@dataclass
class ExecutionResult:
    """Outcome of executing the generated and gold SQL against the DB.

    Always populated alongside the cvc5 verdict so the report can
    show both signals side-by-side. ``status`` is the single-valued
    summary; the booleans give the operator a multiset-vs-set
    distinction (BIRD's official evaluator uses set equality, but
    multiset is strictly stricter and catches duplicate-count bugs).
    """

    status: ExecutionStatus
    """Single-value summary of how execution went."""

    multiset_match: bool = False
    """``True`` iff the two result sets agree as multisets (rows-with-counts)."""

    set_match: bool = False
    """``True`` iff the two result sets agree as sets (deduplicated)."""

    generated_row_count: int = 0
    """Number of rows the generated query returned (0 on error/skipped)."""

    gold_row_count: int = 0
    """Number of rows the gold query returned (0 on error/skipped)."""

    error: str = ""
    """Free-form detail when ``status`` is an error / timeout / unavailable.

    Empty for ``match`` / ``mismatch`` / ``skipped``.
    """


@dataclass
class ForeignKey:
    """A single declared foreign-key edge between two columns.

    Resolved from BIRD's ``{split}_tables.json`` metadata (which encodes
    FKs as integer-index pairs into a flat column list); the loader
    decodes them into table/column names so the rest of the framework
    can use them without re-reading the metadata file.

    ``from_table.from_column`` references ``to_table.to_column``. The
    direction matches BIRD's encoding: every value in ``from_column``
    is required to appear as a value of ``to_column``.
    """

    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class TestCase:
    """A single BIRD record, ready for the runner.

    ``test_case_id`` is ``f"{split}_{question_id}"`` per Req 1.2 and is the
    join key against the Expected_Fail_List and the Run_Manifest.

    ``foreign_keys`` carries the per-database FK list resolved from
    ``{split}_tables.json``. The runner uses it to add referential-
    integrity axioms to the equivalence-check SMT script so a gold
    query that does ``T1 INNER JOIN T2 ON T1.fk = T2.pk`` is provably
    equivalent to the same query without the join, when ``fk → pk``
    is a declared FK. Empty when the metadata file is missing or the
    database has no declared FKs.
    """

    test_case_id: str
    split: str
    db_id: str
    schema: str
    question: str
    evidence: str
    gold_sql: str
    foreign_keys: list[ForeignKey] = field(default_factory=list)


@dataclass
class SkippedTestCase:
    """A BIRD record that could not be turned into a TestCase.

    The loader yields these for missing required fields and missing per-record
    SQLite files (Req 1.7). The suite driver records them as
    ``Verdict.skipped`` Run_Results and continues.
    """

    test_case_id: str
    reason: str


@dataclass
class ConverterError:
    """A structured failure from the SQL_to_DRC_Converter.

    Returned (never raised) by ``bird_benchmark.sql_to_drc.convert_sql`` so
    the suite can keep running after a single Test_Case's gold SQL is out of
    scope.
    """

    kind: Literal["parse_error", "unsupported_feature", "unbound_reference"]
    message: str
    feature: str = ""
    line: int = 1
    column: int = 1


@dataclass
class RunResult:
    """The outcome of running one Test_Case end-to-end.

    Both ``underlying_verdict`` and ``reported_verdict`` are stored so the
    Expected_Fail mask is auditable: ``underlying_verdict`` is what the
    framework actually observed, and ``reported_verdict`` is what the report
    surfaces after applying the override (Req 12.5).
    """

    test_case_id: str
    underlying_verdict: Verdict
    reported_verdict: Verdict
    question: str
    evidence: str
    planner_input: str
    generated_sql: str = ""
    gold_sql: str = ""
    smt_script: str | None = None
    reason: str = ""
    error_code: str = ""
    execution: ExecutionResult | None = None
    """Optional execution-equivalence result.

    Populated when the runner ran both queries against the BIRD
    SQLite DB. ``None`` when execution checks are disabled
    (``RunOptions.execution_check=False``) or when
    :class:`RunResult` is produced via a path before exec is
    sensible (most internal short-circuits set this lazily).
    """


@dataclass
class RunOptions:
    """User-supplied configuration for a single- or suite-mode run.

    ``manifest_path`` / ``report_json_path`` / ``report_md_path`` are
    suite-mode fields (``run_suite`` requires the manifest path; report
    paths default to ``./bird-report.json`` and ``./bird-report.md`` in
    the CLI per the design). Single-mode ignores them.
    """

    bird_root: Path
    split: str
    cvc5_timeout_seconds: int = 30
    per_test_timeout_seconds: int = 60
    expected_fail_path: Path | None = None
    manifest_path: Path | None = None
    resume: bool = False
    cvc5_path: str = "cvc5"
    report_json_path: Path | None = None
    report_md_path: Path | None = None
    execution_check: bool = True
    """Run the generated and gold queries against the BIRD SQLite DB?

    Defaults to ``True`` so a default invocation gets both the cvc5
    verdict and the executed-row-set verdict. The CLI's
    ``--no-execution-check`` flag turns it off when execution would be
    too slow or the SQLite databases aren't installed.
    """

    execution_timeout_seconds: int = 30
    """Per-query wall-clock budget for the execution check.

    Applied independently to the generated and gold queries via
    ``connection.interrupt()`` from a watchdog thread. The default
    matches ``cvc5_timeout_seconds`` so the framework's three timeouts
    (planner, cvc5, executor) start at the same value.
    """


@dataclass
class SingleSelector:
    """Selects exactly one Test_Case from a split for single-test mode.

    Exactly one of ``test_case_id`` / ``question_text`` must be set; the
    Single_Run driver enforces that and exits non-zero otherwise (Req 7.6).
    """

    test_case_id: str | None = None
    question_text: str | None = None


@dataclass
class SuiteSummary:
    """Aggregate output of a suite run.

    ``counts`` is pre-populated with a zero entry for every Verdict so the
    JSON report always contains all nine categories even when none fired
    (Req 9.5).
    """

    counts: dict[Verdict, int] = field(
        default_factory=lambda: {v: 0 for v in Verdict}
    )
    results: list[RunResult] = field(default_factory=list)
    failed_to_write_report: bool = False
    stale_expected_fail_ids: set[str] = field(default_factory=set)


@dataclass
class SampleSummary:
    """Aggregate output of a ``sample`` run.

    Sampling is the "quick look" mode: pick ``n`` Test_Cases from the
    split with a deterministic seed, run each one, and report the
    verdict / execution-equivalence rates plus the cells where the two
    signals disagree. Distinct from :class:`SuiteSummary` because:

    * No manifest, no resume — the sampler is meant for ad-hoc
      diagnostic runs.
    * Tracks both the cvc5 verdict and the execution status side-by-
      side, so the most useful number ("how often does the gold query
      match what the planner builds, by row-set?") is a first-class
      output, not a derived calculation the operator has to do by hand.

    The disagreement counts are the most interesting cells: a high
    ``logical_yes_exec_no`` count means the framework's logical
    equivalence is too strict against BIRD's database (typically
    over-specified gold queries); a high ``logical_no_exec_yes`` count
    is the opposite — the planner's queries return the right rows but
    the framework can't prove the queries are logically equivalent on
    arbitrary inputs.
    """

    seed: int
    """The PRNG seed used to draw the sample. Logged so a re-run can
    reproduce the same selection."""

    requested_count: int
    """The ``--count`` the operator asked for."""

    sampled_count: int
    """How many Test_Cases were actually run.

    May be smaller than ``requested_count`` when the split has fewer
    eligible records (e.g. requested 100 but the split only has 50
    non-skipped records)."""

    verdict_counts: dict[Verdict, int] = field(
        default_factory=lambda: {v: 0 for v in Verdict}
    )
    """Tally of ``RunResult.reported_verdict`` over the sample."""

    execution_counts: dict[ExecutionStatus, int] = field(
        default_factory=lambda: {s: 0 for s in ExecutionStatus}
    )
    """Tally of ``RunResult.execution.status`` over the sample.

    Always populated — ``ExecutionStatus.skipped`` covers the cases
    where execution wasn't run (planner failed, exec disabled).
    """

    multiset_match_count: int = 0
    """Number of sampled cases whose generated and gold queries
    returned the same rows *with the same multiplicity* on the BIRD
    database. Strict subset of ``execution_counts[match]``."""

    logical_yes_exec_no: int = 0
    """Cases where cvc5 said ``equivalent`` but execution said
    ``mismatch`` — usually the planner's query is right on most
    inputs but BIRD's gold query happens to filter rows differently
    on this specific database."""

    logical_no_exec_yes: int = 0
    """Cases where cvc5 said ``not_equivalent`` but execution said
    ``match`` — usually BIRD's gold is over-specified relative to
    the question (extra joins on declared-but-not-enforced FKs, for
    example) and both queries happen to return the same rows on
    BIRD's snapshot."""

    results: list[RunResult] = field(default_factory=list)
    """The raw :class:`RunResult` per Test_Case, in sample order."""

    failed_to_write_report: bool = False
    """Mirrors :attr:`SuiteSummary.failed_to_write_report` so the CLI
    can return the same exit code (3) when reports could not be
    written."""
