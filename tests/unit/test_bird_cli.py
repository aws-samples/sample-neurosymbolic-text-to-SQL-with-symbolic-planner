"""Unit tests for ``bird_benchmark.cli`` (task 14.1).

These tests cover the CLI's three responsibilities: argparse layout
(subcommands, required flags, mutually-exclusive selectors), timeout
validation (Req 6.2), and the exit-code mapping for the dispatch
errors enumerated in the task contract:

* ``0`` — success
* ``1`` — single-mode selector failure (Req 7.4 / 7.5 / 7.6)
* ``2`` — missing/malformed configuration (invalid timeout,
  :class:`BirdLoadError`, :class:`ExpectedFailLoadError`,
  :class:`ManifestParseError`)
* ``3`` — :attr:`SuiteSummary.failed_to_write_report` (Req 9.6)

We drive the CLI by passing an explicit ``argv`` list to
:func:`bird_benchmark.cli.main` and capture stderr in an
:class:`io.StringIO` so the assertions stay decoupled from
:data:`sys.argv` / :data:`sys.stderr`. The dispatch tests
monkeypatch the runner-level callables imported into the
``cli`` module so neither Bedrock nor cvc5 nor the BIRD download
is touched.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import pytest

from bird_benchmark import cli
from bird_benchmark.cli import (
    EXIT_CONFIG_ERROR,
    EXIT_OK,
    EXIT_REPORT_WRITE_ERROR,
    EXIT_SELECTOR_ERROR,
    _build_parser,
    _options_from_args,
    _timeout_int,
    main,
)
from bird_benchmark.expected_fail import ExpectedFailLoadError
from bird_benchmark.loader import BirdLoadError
from bird_benchmark.manifest import ManifestParseError
from bird_benchmark.runner import SingleSelectorError
from bird_benchmark.types import SuiteSummary


# --- _timeout_int -------------------------------------------------------


def test_timeout_int_accepts_in_range_integer():
    """Integer strings inside [1, 3600] round-trip through ``_timeout_int``."""
    assert _timeout_int("1") == 1
    assert _timeout_int("30") == 30
    assert _timeout_int("3600") == 3600


@pytest.mark.parametrize("value", ["0", "-1", "3601", "9999"])
def test_timeout_int_rejects_out_of_range(value):
    """Out-of-range integers are rejected and the error names the value (Req 6.2)."""
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        _timeout_int(value)
    assert value in str(excinfo.value)


@pytest.mark.parametrize("value", ["", "abc", "1.5", "30s", "0x10"])
def test_timeout_int_rejects_non_integer(value):
    """Non-integer strings are rejected and the error names the value (Req 6.2)."""
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        _timeout_int(value)
    assert repr(value) in str(excinfo.value) or value in str(excinfo.value)


# --- argparse layout ----------------------------------------------------


def test_parser_has_single_and_suite_subcommands():
    """Both subcommands are available (Req 10.2)."""
    parser = _build_parser()
    args = parser.parse_args(
        ["single", "--split", "dev", "--bird-root", "/tmp", "--id", "dev_1"]
    )
    assert args.mode == "single"

    args = parser.parse_args(
        [
            "suite",
            "--split",
            "dev",
            "--bird-root",
            "/tmp",
            "--manifest",
            "/tmp/manifest.jsonl",
        ]
    )
    assert args.mode == "suite"


def test_parser_requires_a_subcommand(capsys):
    """Invoking with no subcommand exits non-zero (Req 10.2)."""
    parser = _build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([])
    assert excinfo.value.code != 0


def test_parser_single_requires_split_and_bird_root(capsys):
    """``single`` rejects missing required flags."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["single", "--id", "dev_1"])


def test_parser_suite_requires_manifest(capsys):
    """``suite`` rejects missing ``--manifest``."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["suite", "--split", "dev", "--bird-root", "/tmp"])


def test_parser_default_timeouts_match_design():
    """The default ``--cvc5-timeout`` and ``--per-test-timeout`` match the design."""
    parser = _build_parser()
    args = parser.parse_args(
        ["single", "--split", "dev", "--bird-root", "/tmp", "--id", "dev_1"]
    )
    assert args.cvc5_timeout == 30
    assert args.per_test_timeout == 60


def test_parser_invalid_cvc5_timeout_exits_two(capsys):
    """An out-of-range ``--cvc5-timeout`` triggers argparse's exit-2 path (Req 6.2)."""
    parser = _build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "single",
                "--split",
                "dev",
                "--bird-root",
                "/tmp",
                "--id",
                "dev_1",
                "--cvc5-timeout",
                "9999",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "9999" in err


def test_parser_invalid_per_test_timeout_exits_two(capsys):
    """An out-of-range ``--per-test-timeout`` triggers argparse's exit-2 path (Req 6.2)."""
    parser = _build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "suite",
                "--split",
                "dev",
                "--bird-root",
                "/tmp",
                "--manifest",
                "/tmp/m.jsonl",
                "--per-test-timeout",
                "0",
            ]
        )
    assert excinfo.value.code == 2


def test_parser_non_integer_timeout_exits_two(capsys):
    """A non-integer timeout is rejected with the value named (Req 6.2)."""
    parser = _build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "single",
                "--split",
                "dev",
                "--bird-root",
                "/tmp",
                "--id",
                "dev_1",
                "--cvc5-timeout",
                "abc",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "abc" in err


# --- _options_from_args -------------------------------------------------


def test_options_from_args_single_does_not_set_suite_fields():
    """In ``single`` mode the suite-only ``RunOptions`` fields stay unset."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "single",
            "--split",
            "dev",
            "--bird-root",
            "/tmp/bird",
            "--id",
            "dev_1",
            "--cvc5-timeout",
            "45",
            "--per-test-timeout",
            "120",
            "--cvc5-path",
            "/usr/bin/cvc5",
        ]
    )
    options = _options_from_args(args)
    assert options.split == "dev"
    assert options.bird_root == Path("/tmp/bird")
    assert options.cvc5_timeout_seconds == 45
    assert options.per_test_timeout_seconds == 120
    assert options.cvc5_path == "/usr/bin/cvc5"
    assert options.manifest_path is None
    assert options.resume is False
    assert options.report_json_path is None
    assert options.report_md_path is None


def test_options_from_args_suite_propagates_manifest_resume_and_reports(tmp_path):
    """In ``suite`` mode the manifest, resume flag, and report paths flow into ``RunOptions``."""
    parser = _build_parser()
    manifest = tmp_path / "manifest.jsonl"
    json_report = tmp_path / "report.json"
    md_report = tmp_path / "report.md"
    args = parser.parse_args(
        [
            "suite",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--resume",
            "--report-json",
            str(json_report),
            "--report-md",
            str(md_report),
        ]
    )
    options = _options_from_args(args)
    assert options.manifest_path == manifest
    assert options.resume is True
    assert options.report_json_path == json_report
    assert options.report_md_path == md_report


def test_options_from_args_suite_uses_default_report_paths(tmp_path):
    """When the operator omits the report flags, the design's defaults apply."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "suite",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "m.jsonl"),
        ]
    )
    options = _options_from_args(args)
    assert options.report_json_path == Path("bird-report.json")
    assert options.report_md_path == Path("bird-report.md")


# --- main: dispatch + exit-code mapping --------------------------------


def _stub_async(return_value: Any):
    """Build an ``async`` callable that returns ``return_value``."""

    async def _stub(*args: Any, **kwargs: Any) -> Any:
        return return_value

    return _stub


def _stub_async_raises(exc: BaseException):
    """Build an ``async`` callable that raises ``exc`` when awaited."""

    async def _stub(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return _stub


def test_main_single_success_returns_zero(monkeypatch, tmp_path):
    """A successful single-mode invocation exits with ``EXIT_OK``."""
    captured: dict[str, Any] = {}

    async def _fake_run_single(selector, options, *args, **kwargs):
        captured["selector"] = selector
        captured["options"] = options
        return None

    monkeypatch.setattr(cli, "run_single", _fake_run_single)
    err = io.StringIO()
    code = main(
        [
            "single",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--id",
            "dev_1",
        ],
        stderr=err,
    )
    assert code == EXIT_OK
    assert captured["selector"].test_case_id == "dev_1"
    assert captured["options"].split == "dev"


def test_main_single_selector_error_exits_one(monkeypatch, tmp_path):
    """A :class:`SingleSelectorError` from the runner maps to exit code ``1`` (Req 7.4 / 7.5 / 7.6)."""
    monkeypatch.setattr(
        cli,
        "run_single",
        _stub_async_raises(
            SingleSelectorError(
                "selector test_case_id='dev_999' matched no Test_Case in split 'dev'"
            )
        ),
    )
    err = io.StringIO()
    code = main(
        [
            "single",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--id",
            "dev_999",
        ],
        stderr=err,
    )
    assert code == EXIT_SELECTOR_ERROR
    assert "dev_999" in err.getvalue()


def test_main_single_bird_load_error_exits_two(monkeypatch, tmp_path):
    """A :class:`BirdLoadError` maps to exit code ``2`` (Req 1.5 / 1.6)."""
    monkeypatch.setattr(
        cli,
        "run_single",
        _stub_async_raises(
            BirdLoadError(tmp_path / "dev" / "dev.json", "no such file")
        ),
    )
    err = io.StringIO()
    code = main(
        [
            "single",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--id",
            "dev_1",
        ],
        stderr=err,
    )
    assert code == EXIT_CONFIG_ERROR
    assert "no such file" in err.getvalue()


def test_main_single_expected_fail_load_error_exits_two(monkeypatch, tmp_path):
    """A :class:`ExpectedFailLoadError` maps to exit code ``2`` (Req 12.2)."""
    monkeypatch.setattr(
        cli,
        "run_single",
        _stub_async_raises(
            ExpectedFailLoadError(tmp_path / "missing.txt", "Permission denied")
        ),
    )
    err = io.StringIO()
    code = main(
        [
            "single",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--id",
            "dev_1",
            "--expected-fail",
            str(tmp_path / "missing.txt"),
        ],
        stderr=err,
    )
    assert code == EXIT_CONFIG_ERROR
    assert "Permission denied" in err.getvalue()


def test_main_suite_success_returns_zero(monkeypatch, tmp_path):
    """A clean suite run with ``failed_to_write_report=False`` exits ``EXIT_OK``."""
    monkeypatch.setattr(
        cli, "run_suite", _stub_async(SuiteSummary())
    )
    err = io.StringIO()
    code = main(
        [
            "suite",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "m.jsonl"),
        ],
        stderr=err,
    )
    assert code == EXIT_OK


def test_main_suite_manifest_parse_error_exits_two(monkeypatch, tmp_path):
    """A :class:`ManifestParseError` under ``--resume`` maps to exit code ``2`` (Req 8.5)."""
    monkeypatch.setattr(
        cli,
        "run_suite",
        _stub_async_raises(
            ManifestParseError(
                tmp_path / "m.jsonl", 4, "Expecting value: line 1 column 1"
            )
        ),
    )
    err = io.StringIO()
    code = main(
        [
            "suite",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "m.jsonl"),
            "--resume",
        ],
        stderr=err,
    )
    assert code == EXIT_CONFIG_ERROR
    assert "malformed JSON" in err.getvalue()


def test_main_suite_expected_fail_load_error_exits_two(monkeypatch, tmp_path):
    """A :class:`ExpectedFailLoadError` raised by the suite maps to ``2`` (Req 12.2)."""
    monkeypatch.setattr(
        cli,
        "run_suite",
        _stub_async_raises(
            ExpectedFailLoadError(tmp_path / "missing.txt", "no such file")
        ),
    )
    err = io.StringIO()
    code = main(
        [
            "suite",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "m.jsonl"),
            "--expected-fail",
            str(tmp_path / "missing.txt"),
        ],
        stderr=err,
    )
    assert code == EXIT_CONFIG_ERROR


def test_main_suite_bird_load_error_exits_two(monkeypatch, tmp_path):
    """A :class:`BirdLoadError` from the loader maps to ``2`` (Req 1.5 / 1.6)."""
    monkeypatch.setattr(
        cli,
        "run_suite",
        _stub_async_raises(
            BirdLoadError(tmp_path / "dev" / "dev.json", "missing JSON file")
        ),
    )
    err = io.StringIO()
    code = main(
        [
            "suite",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "m.jsonl"),
        ],
        stderr=err,
    )
    assert code == EXIT_CONFIG_ERROR
    assert "missing JSON file" in err.getvalue()


def test_main_suite_failed_report_write_exits_three(monkeypatch, tmp_path):
    """``SuiteSummary.failed_to_write_report=True`` maps to exit code ``3`` (Req 9.6)."""
    summary = SuiteSummary()
    summary.failed_to_write_report = True
    monkeypatch.setattr(cli, "run_suite", _stub_async(summary))
    err = io.StringIO()
    code = main(
        [
            "suite",
            "--split",
            "dev",
            "--bird-root",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "m.jsonl"),
        ],
        stderr=err,
    )
    assert code == EXIT_REPORT_WRITE_ERROR


# --- __main__ wiring ----------------------------------------------------


def test_main_module_imports_and_calls_cli_main():
    """``bird_benchmark.__main__`` exposes a callable ``main`` that wraps ``cli.main``."""
    import bird_benchmark.__main__ as entry

    # The shim's ``main`` calls ``sys.exit(cli_main())``; we don't actually
    # run it (that would need a full argv), but we can assert the wiring
    # is in place by inspecting the module.
    assert callable(entry.main)
    src = Path(entry.__file__).read_text()
    assert "from bird_benchmark.cli import main" in src
    assert "sys.exit" in src
