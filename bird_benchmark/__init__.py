"""BIRD Benchmark Framework public surface.

The full public surface listed in ``design.md`` is::

    BirdLoader, BirdLoadError, run_single, run_suite, RunOptions,
    Verdict, RunResult, TestCase, ConverterError

Operators surface ``ExpectedFailLoadError`` through the CLI as a non-zero
exit when ``--expected-fail`` points at a missing or unreadable file
(Req 12.2), so it lives on the public surface alongside ``BirdLoadError``.

The ``run_single`` / ``run_suite`` entry points are wired in by tasks 9.1 /
12.1 once ``runner.py`` exists.
"""

from bird_benchmark.expected_fail import ExpectedFailLoadError
from bird_benchmark.installer import (
    InstallError,
    InstallReport,
    install_split,
)
from bird_benchmark.loader import BirdLoader, BirdLoadError
from bird_benchmark.runner import (
    SingleSelectorError,
    run_sample,
    run_single,
    run_suite,
)
from bird_benchmark.types import (
    ConverterError,
    ExecutionResult,
    ExecutionStatus,
    ForeignKey,
    RunOptions,
    RunResult,
    SampleSummary,
    SingleSelector,
    TestCase,
    Verdict,
)

__all__ = [
    "BirdLoader",
    "BirdLoadError",
    "ConverterError",
    "ExecutionResult",
    "ExecutionStatus",
    "ExpectedFailLoadError",
    "ForeignKey",
    "InstallError",
    "InstallReport",
    "RunOptions",
    "RunResult",
    "SampleSummary",
    "SingleSelector",
    "SingleSelectorError",
    "TestCase",
    "Verdict",
    "install_split",
    "run_sample",
    "run_single",
    "run_suite",
]
