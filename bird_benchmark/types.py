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


@dataclass
class TestCase:
    """A single BIRD record, ready for the runner.

    ``test_case_id`` is ``f"{split}_{question_id}"`` per Req 1.2 and is the
    join key against the Expected_Fail_List and the Run_Manifest.
    """

    test_case_id: str
    split: str
    db_id: str
    schema: str
    question: str
    evidence: str
    gold_sql: str


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


@dataclass
class RunOptions:
    """User-supplied configuration for a single- or suite-mode run."""

    bird_root: Path
    split: str
    cvc5_timeout_seconds: int = 30
    per_test_timeout_seconds: int = 60
    expected_fail_path: Path | None = None
    manifest_path: Path | None = None
    resume: bool = False
    cvc5_path: str = "cvc5"


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
