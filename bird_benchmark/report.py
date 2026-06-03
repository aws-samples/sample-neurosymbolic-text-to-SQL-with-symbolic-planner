"""Reporters for Suite_Run results.

This module owns the on-disk shape of the JSON and Markdown reports
described in the design's *Reporters* section. ``write_json_report``
(task 11.1) and ``write_markdown_report`` (task 11.2) live side by
side; the suite driver (task 12.1) calls both at the end of a run.

The JSON shape (Req 9.5) is::

    {
      "summary": {<every Verdict category>: int >= 0},
      "results": [<one entry per Run_Result>]
    }

Every ``Verdict`` category appears in ``summary`` even when zero
Run_Results carry it, and counts are tallied against
``RunResult.reported_verdict`` (the post-Expected_Fail-override verdict)
so the summary mirrors what an operator sees in the markdown report.

The markdown report mirrors that summary with a table and then renders
one section per Verdict category, each section listing every
Run_Result whose ``reported_verdict`` matches. ``not_equivalent`` /
``unknown`` / ``timeout`` sections include the SMT_Script (or an
explicit absence indication) so investigators can replay equivalence
checks; ``converter_failed`` / ``planner_failed`` sections include the
failure ``reason`` instead.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from bird_benchmark.types import RunResult, Verdict


def write_json_report(results: list[RunResult], path: Path) -> None:
    """Write the JSON report for a Suite_Run.

    The output object has exactly two top-level keys: ``summary`` and
    ``results``. ``summary`` maps every Verdict name to a non-negative
    integer count of Run_Results carrying that ``reported_verdict``;
    every Verdict appears, even when its count is zero (Req 9.5).
    ``results`` is the list of ``RunResult`` dataclasses serialised in
    input order, with length equal to ``len(results)`` (Req 9.1).

    The parent directory is created if it does not already exist so a
    fresh ``--report-json`` path under a new directory works without a
    pre-flight ``mkdir``. ``OSError`` / ``IOError`` raised while writing
    propagates to the caller; the suite driver catches it and sets
    ``SuiteSummary.failed_to_write_report`` (Req 9.6, task 12.1).
    """
    counts: dict[str, int] = {v.value: 0 for v in Verdict}
    for r in results:
        # ``reported_verdict`` is the post-override verdict (Req 12.5);
        # ``Verdict`` subclasses ``str`` so ``.value`` is always a key
        # present in ``counts``.
        counts[r.reported_verdict.value] += 1

    payload = {
        "summary": counts,
        "results": [_serialize_run_result(r) for r in results],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default_serializer)


def _serialize_run_result(result: RunResult) -> dict:
    """Convert a ``RunResult`` to a JSON-friendly dict.

    ``Verdict`` is a ``str``-enum so ``asdict`` round-trips its values as
    plain strings; ``smt_script`` stays ``None`` when unset and renders
    as JSON ``null``.
    """
    return asdict(result)


def _default_serializer(obj):
    """Fallback for ``json.dump`` for types it cannot handle natively.

    ``RunResult`` itself does not currently expose ``Path`` or non-string
    ``Enum`` fields, but the fallback keeps the reporter robust against
    future field additions and against nested ``Path`` values that a
    future ``RunResult`` extension might carry.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Cannot serialise {type(obj).__name__}: {obj!r}")


# Per-Verdict rendering policy for the markdown report.
#
# ``not_equivalent`` / ``unknown`` / ``timeout``: the equivalence
# checker either returned a non-equivalent or indeterminate verdict, or
# we hit the cvc5 timeout. Investigators need both SQL strings and the
# SMT script (or the explicit absence indication when the equivalence
# checker did not run) to reproduce the result (Req 9.3).
_VERDICTS_WITH_SMT: frozenset[Verdict] = frozenset(
    {Verdict.not_equivalent, Verdict.unknown, Verdict.timeout}
)

# ``converter_failed`` / ``planner_failed``: there is no SMT script
# because the equivalence checker never ran. Investigators need the
# question, gold SQL, and the failure reason (Req 9.4).
_VERDICTS_WITH_REASON: frozenset[Verdict] = frozenset(
    {Verdict.converter_failed, Verdict.planner_failed}
)

# Indication used when an SMT-bearing verdict has no captured script.
# This happens when the per-Test_Case path short-circuited before the
# equivalence checker could emit one (e.g. an internal cvc5 wrapper
# error for a verdict that still maps to ``unknown``). The text is
# verbatim from the design (Req 9.3).
_SMT_NOT_PRODUCED: str = (
    "_(not produced; equivalence checker did not run)_"
)


def write_markdown_report(results: list[RunResult], path: Path) -> None:
    """Write a Markdown report for a Suite_Run.

    The output starts with a header line that names the split and an
    ISO-8601 UTC timestamp, followed by a ``## Summary`` table that
    lists every Verdict category with its count (Req 9.2). The table
    always includes all nine categories so the report has a stable
    shape even when some verdicts never fired.

    Per-verdict sections follow. Sections are emitted only when at
    least one Run_Result carries that ``reported_verdict`` so empty
    sections do not clutter the report. For each Run_Result in a
    section:

    * ``not_equivalent`` / ``unknown`` / ``timeout`` (Req 9.3): render
      the question, the generated SQL, the gold SQL, and the SMT
      script. When ``smt_script`` is ``None`` (the equivalence checker
      did not run for this Test_Case), substitute the verbatim
      ``"_(not produced; equivalence checker did not run)_"``
      indication.
    * ``converter_failed`` / ``planner_failed`` (Req 9.4): render the
      question, the gold SQL, and the failure ``reason``.
    * Other verdicts (``equivalent``, ``skipped``, ``expected_fail``,
      ``gold_conversion_failure``): render only the ``test_case_id``.
      The summary table already carries the count and the JSON report
      carries the full details, so the markdown stays focused on the
      cases an operator needs to investigate.

    The split is derived from the first Run_Result's ``test_case_id``
    (which the loader builds as ``f"{split}_{question_id}"``, Req 1.2).
    When ``results`` is empty there is no Test_Case to derive the
    split from, so the header reports ``"unknown"``; an empty suite
    still produces a well-formed report with the summary table and no
    per-verdict sections.

    The path's parent directory is created if it does not already
    exist. ``OSError`` / ``IOError`` raised while writing propagates to
    the caller; the suite driver catches it and sets
    ``SuiteSummary.failed_to_write_report`` (Req 9.6, task 12.1).
    """
    # Tally against ``reported_verdict`` so the summary mirrors what
    # the JSON report and the per-section bodies show (Req 12.5).
    counts: dict[Verdict, int] = {v: 0 for v in Verdict}
    for r in results:
        counts[r.reported_verdict] += 1

    split = _derive_split(results)
    timestamp = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append(
        f"# BIRD Benchmark Report — split: {split}, generated: {timestamp}"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("| Verdict | Count |")
    lines.append("| --- | --- |")
    # Iterate the ``Verdict`` enum so the table order is stable and
    # always covers all nine categories, even ones with zero hits.
    for verdict in Verdict:
        lines.append(f"| {verdict.value} | {counts[verdict]} |")
    lines.append("")

    # Bucket the results by ``reported_verdict`` once so each per-
    # verdict section can iterate in input order without re-scanning
    # the full list.
    by_verdict: dict[Verdict, list[RunResult]] = {v: [] for v in Verdict}
    for r in results:
        by_verdict[r.reported_verdict].append(r)

    for verdict in Verdict:
        bucket = by_verdict[verdict]
        if not bucket:
            # Empty sections are omitted (Req 9.3 / 9.4 only require
            # the section to render *when* there are matching
            # Run_Results, and an always-on empty header would make
            # the report noisier than its design template).
            continue
        lines.append(f"## {verdict.value}")
        for r in bucket:
            lines.extend(_render_run_result(r, verdict))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    # Trailing newline keeps the file POSIX-clean and matches what
    # editors expect; ``"\n".join(...) + "\n"`` is cheaper than
    # ``writelines`` here because the lines are short.
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if not lines or lines[-1] != "":
            f.write("\n")


def _derive_split(results: list[RunResult]) -> str:
    """Recover the split name from a Run_Result's Test_Case_ID.

    The loader builds ``test_case_id`` as ``f"{split}_{question_id}"``
    (Req 1.2), so splitting on the first ``_`` recovers the split
    string. When ``results`` is empty we have nothing to derive from
    and fall back to ``"unknown"``; the suite driver still gets a
    well-formed report it can write.
    """
    if not results:
        return "unknown"
    head = results[0].test_case_id.split("_", 1)
    return head[0] if head and head[0] else "unknown"


def _render_run_result(result: RunResult, verdict: Verdict) -> list[str]:
    """Render a single Run_Result inside its per-verdict section.

    The section header (``## {verdict}``) is emitted by the caller; this
    helper produces the per-Run_Result lines including the
    ``### {test_case_id}`` sub-header.
    """
    lines: list[str] = [f"### {result.test_case_id}"]

    if verdict in _VERDICTS_WITH_SMT:
        # Investigation cases: surface enough context to replay the
        # equivalence check (Req 9.3).
        lines.append(f"**Question:** {result.question}")
        lines.append("**Generated SQL:**")
        lines.append("```sql")
        lines.append(result.generated_sql)
        lines.append("```")
        lines.append("**Gold SQL:**")
        lines.append("```sql")
        lines.append(result.gold_sql)
        lines.append("```")
        if result.smt_script is None:
            lines.append(f"**SMT script:** {_SMT_NOT_PRODUCED}")
        else:
            lines.append("**SMT script:**")
            lines.append("```smt2")
            lines.append(result.smt_script)
            lines.append("```")
    elif verdict in _VERDICTS_WITH_REASON:
        # Failure cases that never reached the equivalence checker
        # (Req 9.4): no SMT script exists, the reason is what matters.
        lines.append(f"**Question:** {result.question}")
        lines.append("**Gold SQL:**")
        lines.append("```sql")
        lines.append(result.gold_sql)
        lines.append("```")
        lines.append(f"**Reason:** {result.reason}")
    # Other verdicts (equivalent / skipped / expected_fail /
    # gold_conversion_failure) only get the Test_Case_ID. The summary
    # table carries the count, and the JSON report carries the full
    # detail for any deeper inspection.

    return lines
