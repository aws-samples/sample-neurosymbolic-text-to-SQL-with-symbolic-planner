"""Unit tests for the BIRD ``sample`` subcommand.

The sampler is the diagnostic verb between ``single`` (one Test_Case)
and ``suite`` (the whole split). It draws ``count`` Test_Cases at
random with a deterministic seed and reports both the cvc5 logical-
equivalence rate and the SQLite execution-equivalence rate, plus the
disagreement cells.

These tests cover:

1. End-to-end: stub planner / converter / equivalence checker /
   executor → the sampler runs the right number of Test_Cases,
   tallies counts correctly, identifies disagreement cells.
2. Determinism: same seed → same selection.
3. Edge cases: empty split, requested > eligible, validation
   errors on count / seed.
4. CLI plumbing: the new ``sample`` subparser parses, dispatches to
   :func:`run_sample`, and maps validation errors to
   ``EXIT_CONFIG_ERROR``.
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pytest

from bird_benchmark import cli
from bird_benchmark.cli import EXIT_CONFIG_ERROR, EXIT_OK
from bird_benchmark.runner import run_sample
from bird_benchmark.types import (
    ExecutionResult,
    ExecutionStatus,
    RunOptions,
    SampleSummary,
    SkippedTestCase,
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
# Helpers
# ---------------------------------------------------------------------------


def _make_drc(name: str = "x", relation: str = "T") -> DRCExpression:
    return DRCExpression(
        result_variables=[ColumnVariable(name=name)],
        condition=MembershipNode(variables=[name], relation=relation),
    )


def _make_planner_success(sql: str = "SELECT id FROM t") -> TextToSQLSuccess:
    drc = _make_drc()
    tree = OperationTree(root=TableLeafNode(table_name="t"))
    return TextToSQLSuccess(
        sql=sql,
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


def _equivalence_returning(result):
    async def _check(*args, **kwargs):
        return result

    return _check


def _exec_returning(result: ExecutionResult):
    def _exec(**kwargs):
        return result

    return _exec


def _make_test_case(idx: int) -> TestCase:
    return TestCase(
        test_case_id=f"dev_{idx}",
        split="dev",
        db_id="t",
        schema="CREATE TABLE t (id INT)",
        question=f"q{idx}",
        evidence="",
        gold_sql="SELECT id FROM t",
    )


def _make_loader(items: list[Any]):
    """Return a loader_factory that yields ``items`` in order."""

    class _Loader:
        def __init__(self, *_args, **_kwargs):
            pass

        def load(self) -> Iterable[Any]:
            yield from items

    def _factory(_config):
        return _Loader()

    return _factory


def _make_options(tmp_path: Path) -> RunOptions:
    return RunOptions(
        bird_root=tmp_path / "bird",
        split="dev",
        cvc5_timeout_seconds=30,
        per_test_timeout_seconds=60,
    )


# ---------------------------------------------------------------------------
# Determinism: same seed → same selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sample_with_same_seed_picks_same_test_cases(tmp_path):
    """Two runs with identical seed + count + split produce identical samples."""

    cases = [_make_test_case(i) for i in range(20)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    out = io.StringIO()

    summary_a = await run_sample(
        options,
        count=5,
        seed=42,
        expected_fail=set(),
        stdout=out,
        loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(
                status=ExecutionStatus.match,
                set_match=True,
                multiset_match=True,
            )
        ),
    )

    summary_b = await run_sample(
        options,
        count=5,
        seed=42,
        expected_fail=set(),
        stdout=io.StringIO(),
        loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(
                status=ExecutionStatus.match,
                set_match=True,
                multiset_match=True,
            )
        ),
    )

    ids_a = [r.test_case_id for r in summary_a.results]
    ids_b = [r.test_case_id for r in summary_b.results]
    assert ids_a == ids_b


@pytest.mark.asyncio
async def test_run_sample_with_different_seeds_picks_different_test_cases(tmp_path):
    """Different seeds produce different selections (sanity: not always equal)."""

    cases = [_make_test_case(i) for i in range(50)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)

    a = await run_sample(
        options, count=5, seed=1,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )
    b = await run_sample(
        options, count=5, seed=999,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    ids_a = [r.test_case_id for r in a.results]
    ids_b = [r.test_case_id for r in b.results]
    # Both runs must produce 5 results; in the 50-case pool the
    # probability of identical selections under different seeds is
    # negligible.
    assert ids_a != ids_b


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sample_records_logical_equivalence_rate(tmp_path):
    """Verdict counts and the equivalent-rate match the stub's behaviour."""

    cases = [_make_test_case(i) for i in range(10)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)

    summary = await run_sample(
        options, count=10, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        # Every case → cvc5 says equivalent.
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(
                status=ExecutionStatus.match,
                set_match=True, multiset_match=True,
            )
        ),
    )

    assert summary.sampled_count == 10
    assert summary.verdict_counts[Verdict.equivalent] == 10
    assert summary.execution_counts[ExecutionStatus.match] == 10
    assert summary.multiset_match_count == 10
    # No disagreements when both signals say "yes" everywhere.
    assert summary.logical_yes_exec_no == 0
    assert summary.logical_no_exec_yes == 0


@pytest.mark.asyncio
async def test_run_sample_counts_logical_no_exec_yes_disagreement(tmp_path):
    """The dev_1519 disagreement cell: cvc5 says no, exec says match.

    This is the highest-signal cell — it surfaces over-specified gold
    queries (extra joins, redundant filters) that don't change the
    answer on BIRD's snapshot.
    """

    cases = [_make_test_case(i) for i in range(8)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)

    summary = await run_sample(
        options, count=8, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        # Every case: cvc5 says NOT equivalent, exec says match.
        check_equivalence_callable=_equivalence_returning(NotEquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(
                status=ExecutionStatus.match,
                set_match=True, multiset_match=True,
            )
        ),
    )

    assert summary.verdict_counts[Verdict.not_equivalent] == 8
    assert summary.execution_counts[ExecutionStatus.match] == 8
    assert summary.logical_no_exec_yes == 8
    assert summary.logical_yes_exec_no == 0


@pytest.mark.asyncio
async def test_run_sample_counts_logical_yes_exec_no_disagreement(tmp_path):
    """The opposite cell: cvc5 says yes, exec says rows differ."""

    cases = [_make_test_case(i) for i in range(4)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)

    summary = await run_sample(
        options, count=4, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(
                status=ExecutionStatus.mismatch,
                set_match=False, multiset_match=False,
            )
        ),
    )
    assert summary.logical_yes_exec_no == 4


@pytest.mark.asyncio
async def test_run_sample_summary_emits_aggregate_lines_to_stdout(tmp_path):
    """The headline rates show up in stdout for the operator."""

    cases = [_make_test_case(i) for i in range(5)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    out = io.StringIO()

    await run_sample(
        options, count=5, seed=7,
        expected_fail=set(), stdout=out, loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(status=ExecutionStatus.match, set_match=True)
        ),
    )

    output = out.getvalue()
    assert "# Sample Summary" in output
    assert "Logically equivalent (cvc5):" in output
    assert "Execution-equivalent (set match):" in output
    assert "Disagreement (logical=yes, exec=no):" in output
    assert "Disagreement (logical=no, exec=yes):" in output
    # The seed should be surfaced so a re-run can reproduce.
    assert "seed=7" in output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sample_handles_count_larger_than_available(tmp_path):
    """When count > eligible, the sampler runs every eligible case once."""

    cases = [_make_test_case(i) for i in range(3)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)

    summary = await run_sample(
        options, count=100, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    assert summary.requested_count == 100
    assert summary.sampled_count == 3
    assert len(summary.results) == 3


@pytest.mark.asyncio
async def test_run_sample_skips_loader_skipped_records(tmp_path):
    """Loader-side skips don't enter the sampling pool."""

    items = [
        SkippedTestCase(test_case_id="dev_record_0", reason="missing question"),
        _make_test_case(1),
        SkippedTestCase(test_case_id="dev_record_2", reason="missing SQL"),
        _make_test_case(3),
    ]
    loader = _make_loader(items)
    options = _make_options(tmp_path)

    summary = await run_sample(
        options, count=2, seed=0,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    # Only the two non-skipped Test_Cases are eligible; both are run.
    sampled_ids = {r.test_case_id for r in summary.results}
    assert sampled_ids == {"dev_1", "dev_3"}


@pytest.mark.asyncio
async def test_run_sample_handles_empty_split(tmp_path):
    """An empty split surfaces a non-fatal warning; sampled_count is 0."""

    loader = _make_loader([])
    options = _make_options(tmp_path)
    err = io.StringIO()

    summary = await run_sample(
        options, count=5, seed=0,
        expected_fail=set(),
        stdout=io.StringIO(),
        stderr=err,
        loader_factory=loader,
    )

    assert summary.sampled_count == 0
    assert summary.results == []
    assert "no eligible Test_Cases" in err.getvalue()


@pytest.mark.asyncio
async def test_run_sample_rejects_negative_count(tmp_path):
    """``count`` must be ≥ 1."""
    options = _make_options(tmp_path)
    with pytest.raises(ValueError):
        await run_sample(
            options, count=0, seed=0,
            expected_fail=set(),
            stdout=io.StringIO(),
            loader_factory=_make_loader([]),
        )


@pytest.mark.asyncio
async def test_run_sample_rejects_negative_seed(tmp_path):
    """``seed`` must be ≥ 0."""
    options = _make_options(tmp_path)
    with pytest.raises(ValueError):
        await run_sample(
            options, count=1, seed=-1,
            expected_fail=set(),
            stdout=io.StringIO(),
            loader_factory=_make_loader([]),
        )


# ---------------------------------------------------------------------------
# CLI: --count, --seed, dispatch
# ---------------------------------------------------------------------------


def test_cli_sample_subparser_requires_count_and_seed():
    """Argparse refuses missing required flags."""
    with pytest.raises(SystemExit):
        cli.main(
            ["sample", "--split", "dev", "--bird-root", "/tmp"],
            stderr=io.StringIO(), stdout=io.StringIO(),
        )


def test_cli_sample_options_carry_count_and_seed(tmp_path):
    """``--count`` / ``--seed`` / ``--output-dir`` parse into the args namespace."""
    from bird_benchmark.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "sample",
            "--split", "dev",
            "--bird-root", "/tmp/bird",
            "--count", "25",
            "--seed", "42",
            "--output-dir", str(tmp_path / "sample-out"),
        ]
    )
    assert args.mode == "sample"
    assert args.count == 25
    assert args.seed == 42
    assert args.output_dir == tmp_path / "sample-out"


def test_cli_sample_subparser_makes_output_dir_optional(tmp_path):
    """``--output-dir`` is now optional; the CLI auto-allocates ``~/runs/run-NN``."""
    parser = __import__(
        "bird_benchmark.cli", fromlist=["_build_parser"]
    )._build_parser()
    args = parser.parse_args(
        [
            "sample",
            "--split", "dev",
            "--bird-root", "/tmp/bird",
            "--count", "1",
            "--seed", "0",
        ]
    )
    # Argparse leaves the field as None when the operator omits it;
    # _run_sample is responsible for materialising the default.
    assert args.output_dir is None


def test_allocate_default_runs_dir_picks_run_01_in_empty_parent(tmp_path):
    """An empty parent directory yields ``run-01``."""
    from bird_benchmark.cli import _allocate_default_runs_dir

    runs = tmp_path / "runs"
    out = _allocate_default_runs_dir(parent=runs)
    assert out == runs / "run-01"
    assert out.is_dir()


def test_allocate_default_runs_dir_picks_max_plus_one(tmp_path):
    """The next ``run-NN`` is always one above the highest existing index."""
    from bird_benchmark.cli import _allocate_default_runs_dir

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run-01").mkdir()
    (runs / "run-05").mkdir()
    (runs / "run-12").mkdir()

    out = _allocate_default_runs_dir(parent=runs)
    assert out == runs / "run-13"
    assert out.is_dir()


def test_allocate_default_runs_dir_ignores_non_integer_suffixes(tmp_path):
    """Operator-named ``run-foo`` directories don't bump the counter."""
    from bird_benchmark.cli import _allocate_default_runs_dir

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run-foo").mkdir()
    (runs / "run-2024-01-01").mkdir()
    (runs / "scratch").mkdir()  # no run- prefix at all
    (runs / "run-03").mkdir()

    out = _allocate_default_runs_dir(parent=runs)
    # Only ``run-03`` counts; the others are ignored.
    assert out == runs / "run-04"


def test_allocate_default_runs_dir_creates_parent(tmp_path):
    """Parent directory is created on demand, including nested paths."""
    from bird_benchmark.cli import _allocate_default_runs_dir

    runs = tmp_path / "deep" / "nested" / "runs"
    assert not runs.exists()
    out = _allocate_default_runs_dir(parent=runs)
    assert runs.is_dir()
    assert out.parent == runs


def test_allocate_default_runs_dir_handles_three_digit_indexes(tmp_path):
    """Indexes above 99 are not zero-padded; the suffix preserves their width."""
    from bird_benchmark.cli import _allocate_default_runs_dir

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run-150").mkdir()

    out = _allocate_default_runs_dir(parent=runs)
    # Padding is "at least 2"; 151 already exceeds two digits so the
    # suffix is rendered as-is.
    assert out == runs / "run-151"


def test_cli_run_sample_uses_auto_dir_when_omitted(monkeypatch, tmp_path):
    """End-to-end: omitting ``--output-dir`` writes to an auto-allocated dir."""
    from bird_benchmark import cli as cli_mod

    runs_root = tmp_path / "runs"

    # Monkeypatch the default runs parent so the test doesn't touch
    # the real ~/runs directory.
    monkeypatch.setattr(cli_mod, "_DEFAULT_RUNS_PARENT", runs_root)

    captured: dict[str, Any] = {}

    async def _fake_sample(*_args, output_dir=None, **_kwargs):
        captured["output_dir"] = output_dir
        # Make sure the directory really exists at the time
        # ``run_sample`` is invoked.
        captured["exists_at_call"] = output_dir.is_dir()
        return SampleSummary(seed=0, requested_count=1, sampled_count=0)

    monkeypatch.setattr(cli_mod, "run_sample", _fake_sample)

    code = cli.main(
        [
            "sample",
            "--split", "dev",
            "--bird-root", str(tmp_path),
            "--count", "1",
            "--seed", "0",
        ],
        stderr=io.StringIO(),
        stdout=io.StringIO(),
    )

    assert code == EXIT_OK
    assert captured["output_dir"] == runs_root / "run-01"
    assert captured["exists_at_call"] is True


def test_cli_sample_options_does_not_set_runoptions_report_paths(tmp_path):
    """Sample mode no longer wires ``--report-json``/``--report-md`` flags."""
    from bird_benchmark.cli import _build_parser, _options_from_args

    parser = _build_parser()
    args = parser.parse_args(
        [
            "sample",
            "--split", "dev", "--bird-root", "/tmp/bird",
            "--count", "1", "--seed", "0",
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    options = _options_from_args(args)
    # ``RunOptions.report_*`` is owned by ``suite``; sample writes its
    # own files under ``--output-dir`` instead.
    assert options.report_json_path is None
    assert options.report_md_path is None


def test_cli_sample_invalid_count_maps_to_config_error(monkeypatch, tmp_path):
    """A ``ValueError`` from run_sample maps to ``EXIT_CONFIG_ERROR``."""
    # The argparse layer accepts any int, so a count of 0 reaches
    # run_sample, which raises ValueError. The CLI catches it and
    # surfaces ``EXIT_CONFIG_ERROR``.
    from bird_benchmark import cli as cli_mod

    async def _fake_sample(*_args, **_kwargs):
        raise ValueError("--count must be a positive integer, got 0")

    monkeypatch.setattr(cli_mod, "run_sample", _fake_sample)

    err = io.StringIO()
    code = cli.main(
        [
            "sample",
            "--split", "dev", "--bird-root", str(tmp_path),
            "--count", "0", "--seed", "0",
            "--output-dir", str(tmp_path / "out"),
        ],
        stderr=err, stdout=io.StringIO(),
    )
    assert code == EXIT_CONFIG_ERROR
    assert "--count must be" in err.getvalue()


# ---------------------------------------------------------------------------
# Output directory mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sample_writes_per_case_files(tmp_path):
    """``output_dir`` mode writes one transcript per Test_Case."""

    cases = [_make_test_case(i) for i in range(3)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    output_dir = tmp_path / "sample-out"

    summary = await run_sample(
        options, count=3, seed=0,
        output_dir=output_dir,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    # One file per sampled Test_Case.
    transcripts = sorted(output_dir.glob("dev_*.md"))
    assert len(transcripts) == 3
    # Each transcript starts with the BIRD context banner.
    for path in transcripts:
        body = path.read_text()
        assert "# BIRD Test_Case" in body
        assert "# Run_Result" in body

    # The summary files exist alongside.
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "summary.json").is_file()
    assert summary.failed_to_write_report is False


@pytest.mark.asyncio
async def test_run_sample_summary_md_lists_test_case_ids_per_bucket(tmp_path):
    """``summary.md`` lists the actual IDs in each verdict bucket as links."""

    cases = [_make_test_case(i) for i in range(4)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    output_dir = tmp_path / "sample-out"

    # Mix of equivalent + not_equivalent verdicts.
    eq_count = {"value": 0}

    async def _alternating_check(*args, **kwargs):
        eq_count["value"] += 1
        if eq_count["value"] % 2 == 1:
            return EquivalentResult()
        return NotEquivalentResult()

    await run_sample(
        options, count=4, seed=0,
        output_dir=output_dir,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_alternating_check,
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    summary_md = (output_dir / "summary.md").read_text()
    # The verdict-breakdown section should name each ID and link it to
    # the per-case file.
    assert "## Verdict breakdown" in summary_md
    # Two cases land in equivalent, two in not_equivalent.
    assert "`equivalent` — 2" in summary_md
    assert "`not_equivalent` — 2" in summary_md
    # Linked IDs use relative ``./{id}.md`` paths.
    for tid in ("dev_0", "dev_1", "dev_2", "dev_3"):
        # The relative-link rendering must appear at least once for
        # each sampled ID.
        assert f"[{tid}](./{tid}.md)" in summary_md


@pytest.mark.asyncio
async def test_run_sample_summary_md_lists_disagreement_ids(tmp_path):
    """The ``logical=no, exec=yes`` cell lists the offending IDs."""

    cases = [_make_test_case(i) for i in range(2)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    output_dir = tmp_path / "sample-out"

    await run_sample(
        options, count=2, seed=0,
        output_dir=output_dir,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        # Both cases: cvc5 says no, exec says yes — the dev_1519 cell.
        check_equivalence_callable=_equivalence_returning(NotEquivalentResult()),
        exec_callable=_exec_returning(
            ExecutionResult(status=ExecutionStatus.match, set_match=True)
        ),
    )

    summary_md = (output_dir / "summary.md").read_text()
    assert "logical=no, exec=yes:** 2" in summary_md
    # Both IDs surface as links in that bucket.
    assert "[dev_0](./dev_0.md)" in summary_md
    assert "[dev_1](./dev_1.md)" in summary_md


@pytest.mark.asyncio
async def test_run_sample_summary_json_carries_id_lists_per_bucket(tmp_path):
    """``summary.json`` matches the markdown structure for machine consumers."""

    cases = [_make_test_case(i) for i in range(3)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    output_dir = tmp_path / "sample-out"

    await run_sample(
        options, count=3, seed=0,
        output_dir=output_dir,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    payload = json.loads((output_dir / "summary.json").read_text())
    assert payload["sampled_count"] == 3
    assert payload["seed"] == 0
    assert payload["rates"]["logical_equivalent"]["count"] == 3
    assert payload["rates"]["execution_match"]["count"] == 3
    # Verdict breakdown carries IDs per bucket.
    eq_bucket = payload["verdict_breakdown"]["equivalent"]
    assert eq_bucket["count"] == 3
    assert sorted(eq_bucket["ids"]) == ["dev_0", "dev_1", "dev_2"]


@pytest.mark.asyncio
async def test_run_sample_directory_mode_writes_progress_to_stdout(tmp_path):
    """In directory mode, stdout gets one progress line per case + a headline."""

    cases = [_make_test_case(i) for i in range(2)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    output_dir = tmp_path / "sample-out"
    out = io.StringIO()

    await run_sample(
        options, count=2, seed=0,
        output_dir=output_dir,
        expected_fail=set(), stdout=out, loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    output = out.getvalue()
    # One progress line per case (order is determined by the seed,
    # so we don't pin specific positions — both must appear).
    assert "dev_0 verdict=equivalent exec=match" in output
    assert "dev_1 verdict=equivalent exec=match" in output
    assert output.count("verdict=equivalent exec=match") == 2
    # Headline summary at the end with the pointer to summary.md.
    assert "sample summary:" in output
    assert "logical-equivalent=2" in output
    assert "summary written to" in output
    # The transcripts must NOT be written to stdout.
    assert "# Run_Result" not in output
    assert "# BIRD Test_Case" not in output


@pytest.mark.asyncio
async def test_run_sample_directory_mode_creates_missing_dir(tmp_path):
    """``run_sample`` creates the output directory if it doesn't exist."""

    cases = [_make_test_case(0)]
    loader = _make_loader(cases)
    options = _make_options(tmp_path)
    output_dir = tmp_path / "deep" / "nested" / "sample-out"
    assert not output_dir.exists()

    await run_sample(
        options, count=1, seed=0,
        output_dir=output_dir,
        expected_fail=set(), stdout=io.StringIO(), loader_factory=loader,
        planner_callable=_planner_returning(_make_planner_success()),
        convert_sql_callable=_converter_returning(_make_drc()),
        check_equivalence_callable=_equivalence_returning(EquivalentResult()),
        exec_callable=_exec_returning(ExecutionResult(status=ExecutionStatus.match)),
    )

    assert output_dir.is_dir()
    assert (output_dir / "summary.md").is_file()
