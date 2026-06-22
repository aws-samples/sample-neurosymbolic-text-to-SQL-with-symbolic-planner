"""Unit tests for ``bird_benchmark.installer`` and the ``install`` CLI.

The installer exists to turn the BIRD distribution into the on-disk
layout :class:`bird_benchmark.loader.BirdLoader` expects. These tests
cover three slices of that contract:

1. **Layout normalisation**: BIRD has shipped at least three archive
   shapes in production (direct, dated wrapper, nested
   ``{split}_databases.zip``). The installer must turn each of them
   into the same canonical layout.
2. **Failure modes**: invalid URL schemes, truncated downloads,
   zip-slip entries, and unknown splits must all be rejected with a
   :class:`InstallError` whose ``stage`` matches the contract.
3. **CLI plumbing**: ``bird-benchmark install`` must dispatch the
   right arguments, surface success on stdout, and map
   :class:`InstallError` to ``EXIT_CONFIG_ERROR``.

We never touch the network. Every test serves a synthesised ZIP via a
``file://`` URL through the installer's ``url_opener`` test seam, or
constructs a stub callable that returns a pre-canned response. The
SQLite databases are real but tiny — one ``CREATE TABLE`` per file —
because the verify step actually opens them with :mod:`sqlite3`.
"""

from __future__ import annotations

import io
import json
import sqlite3
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import pytest

from bird_benchmark import cli
from bird_benchmark.cli import (
    EXIT_CONFIG_ERROR,
    EXIT_OK,
)
from bird_benchmark.installer import (
    InstallError,
    InstallReport,
    default_url_for,
    install_split,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_sqlite(path: Path, db_id: str) -> None:
    """Create a tiny SQLite file with one CREATE TABLE statement.

    The loader only reads ``sqlite_master`` for ``CREATE TABLE``
    statements, so any non-empty schema satisfies the verify step. We
    use a deliberately minimal table so the test fixture stays fast
    even when the test suite runs hundreds of times in CI.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        # db_id is a test-controlled identifier; validate it's alphanumeric.
        table_name = f"{db_id}_main"
        if not all(c.isalnum() or c == "_" for c in table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query
        conn.execute(
            "CREATE TABLE [" + table_name + "] (id INTEGER PRIMARY KEY, value TEXT)"
        )
        conn.commit()


def _make_record(question_id: int, db_id: str, *, split: str) -> dict:
    """Synthesise one BIRD JSON record with all required fields."""

    return {
        "question_id": question_id,
        "db_id": db_id,
        "question": f"Synthetic question {question_id} for {split}",
        "evidence": "",
        "SQL": f"SELECT id FROM {db_id}_main",
    }


def _build_canonical_archive(
    archive_path: Path,
    split: str,
    *,
    inner_dir: str | None = None,
    embed_databases_zip: bool = False,
    extra_entries: Iterable[tuple[str, bytes]] = (),
) -> None:
    """Build a synthetic BIRD ZIP for a single split.

    Parameters
    ----------
    archive_path:
        Where to write the ZIP.
    split:
        The split name (``"dev"`` etc.). Drives the JSON filename and
        the databases directory name.
    inner_dir:
        If set, wrap every entry in this prefix. ``"dev_20240627"``
        reproduces the dated-wrapper layout BIRD has shipped.
    embed_databases_zip:
        If True, ship the per-database SQLite files as a nested
        ``{split}_databases.zip`` inside the split directory rather
        than as loose files. Reproduces the third on-disk shape.
    extra_entries:
        Extra ``(arcname, raw_bytes)`` entries to add to the archive.
        Used by the zip-slip test to inject malicious entry names.
    """

    db_id = "synth_db"
    record = _make_record(0, db_id, split=split)

    # Build the SQLite file in a scratch directory; we'll read its
    # bytes back when adding it to the archive.
    scratch = archive_path.parent / "_scratch"
    if scratch.exists():
        # Each fixture builder gets a fresh scratch dir.
        for child in scratch.rglob("*"):
            if child.is_file():
                child.unlink()
        scratch.rmdir()
    scratch.mkdir(parents=True, exist_ok=True)
    sqlite_path = scratch / f"{db_id}.sqlite"
    _make_sqlite(sqlite_path, db_id)
    sqlite_bytes = sqlite_path.read_bytes()
    json_bytes = json.dumps([record]).encode("utf-8")

    # Compute archive-internal paths relative to ``inner_dir``.
    prefix = f"{inner_dir}/" if inner_dir else ""

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(f"{prefix}{split}.json", json_bytes)
        if embed_databases_zip:
            # Build the nested databases archive in memory.
            inner_buffer = io.BytesIO()
            with zipfile.ZipFile(inner_buffer, "w") as inner_zf:
                inner_zf.writestr(
                    f"{split}_databases/{db_id}/{db_id}.sqlite",
                    sqlite_bytes,
                )
            zf.writestr(
                f"{prefix}{split}_databases.zip", inner_buffer.getvalue()
            )
        else:
            zf.writestr(
                f"{prefix}{split}_databases/{db_id}/{db_id}.sqlite",
                sqlite_bytes,
            )
        for arcname, raw in extra_entries:
            zf.writestr(arcname, raw)


def _file_url_opener(archive_path: Path):
    """Return a ``url_opener`` callable that serves ``archive_path``.

    The installer accepts ``file://`` URLs natively, but we still go
    through the test-seam so we can simulate a missing ``Content-Length``
    header (some BIRD mirrors omit it) and so the installer's
    ``url_opener`` parameter is exercised end-to-end.
    """

    def open_url(url: str) -> urllib.request.addinfourl:
        # Strip the scheme; we read the file directly.
        parsed = urlparse(url)
        target = Path(parsed.path)
        data = target.read_bytes()
        # Build a real ``addinfourl`` so the installer's
        # ``response.headers.get("Content-Length")`` lookup works.
        from email.message import Message

        headers = Message()
        headers["Content-Length"] = str(len(data))
        return urllib.request.addinfourl(  # type: ignore[arg-type]
            io.BytesIO(data),
            headers,
            url,
            200,
        )

    return open_url


# ---------------------------------------------------------------------------
# default_url_for
# ---------------------------------------------------------------------------


def test_default_url_for_known_splits():
    """Both publicly-distributed splits resolve to the official endpoint."""
    assert default_url_for("dev").endswith("/dev.zip")
    assert default_url_for("train").endswith("/train.zip")


def test_default_url_for_unknown_split_raises_validate():
    """Unknown splits surface an ``InstallError`` at the validate stage."""
    with pytest.raises(InstallError) as excinfo:
        default_url_for("test")
    assert excinfo.value.stage == "validate"
    assert "test" in excinfo.value.message


# ---------------------------------------------------------------------------
# install_split — happy path and layout variants
# ---------------------------------------------------------------------------


def test_install_split_canonical_layout(tmp_path):
    """Direct archive: JSON + databases/ at the top level."""
    archive = tmp_path / "src" / "dev.zip"
    archive.parent.mkdir(parents=True)
    _build_canonical_archive(archive, "dev")

    bird_root = tmp_path / "bird"
    report = install_split(
        bird_root=bird_root,
        split="dev",
        url=archive.as_uri(),
        url_opener=_file_url_opener(archive),
    )

    assert isinstance(report, InstallReport)
    assert report.split == "dev"
    assert report.json_path == bird_root / "dev" / "dev.json"
    assert report.databases_dir == bird_root / "dev" / "dev_databases"
    assert report.database_count == 1
    assert report.test_case_count == 1
    assert report.skipped_records == 0
    assert report.download_bytes > 0


def test_install_split_dated_wrapper_layout(tmp_path):
    """Dated-wrapper archive (``dev_20240627/...``) is flattened."""
    archive = tmp_path / "src" / "dev.zip"
    archive.parent.mkdir(parents=True)
    _build_canonical_archive(archive, "dev", inner_dir="dev_20240627")

    bird_root = tmp_path / "bird"
    report = install_split(
        bird_root=bird_root,
        split="dev",
        url=archive.as_uri(),
        url_opener=_file_url_opener(archive),
    )

    assert (bird_root / "dev" / "dev.json").is_file()
    assert (bird_root / "dev" / "dev_databases" / "synth_db" / "synth_db.sqlite").is_file()
    # The installer should mention the flattening in its notes so the
    # CLI's success summary explains the rename to operators.
    assert any("flattened" in note for note in report.notes)


def test_install_split_nested_databases_zip(tmp_path):
    """Nested ``{split}_databases.zip`` is extracted in place."""
    archive = tmp_path / "src" / "dev.zip"
    archive.parent.mkdir(parents=True)
    _build_canonical_archive(archive, "dev", embed_databases_zip=True)

    bird_root = tmp_path / "bird"
    report = install_split(
        bird_root=bird_root,
        split="dev",
        url=archive.as_uri(),
        url_opener=_file_url_opener(archive),
    )

    # The nested zip should have been extracted and removed.
    assert not (bird_root / "dev" / "dev_databases.zip").exists()
    assert (bird_root / "dev" / "dev_databases" / "synth_db" / "synth_db.sqlite").is_file()
    assert any("nested" in note for note in report.notes)


def test_install_split_short_circuits_when_already_installed(tmp_path):
    """A second install on a valid layout skips download (Req: idempotent)."""
    archive = tmp_path / "src" / "dev.zip"
    archive.parent.mkdir(parents=True)
    _build_canonical_archive(archive, "dev")

    bird_root = tmp_path / "bird"
    install_split(
        bird_root=bird_root,
        split="dev",
        url=archive.as_uri(),
        url_opener=_file_url_opener(archive),
    )

    # Replace the URL opener with one that fails loudly to prove the
    # second call doesn't re-download.
    def fail(_url: str):
        raise AssertionError("install_split must not re-download on short-circuit")

    report = install_split(
        bird_root=bird_root,
        split="dev",
        url=archive.as_uri(),
        url_opener=fail,
    )
    assert report.download_bytes == 0
    assert any("short-circuit" in note for note in report.notes)


def test_install_split_force_redownloads(tmp_path):
    """``force=True`` re-downloads even on a valid layout."""
    archive = tmp_path / "src" / "dev.zip"
    archive.parent.mkdir(parents=True)
    _build_canonical_archive(archive, "dev")

    bird_root = tmp_path / "bird"
    install_split(
        bird_root=bird_root,
        split="dev",
        url=archive.as_uri(),
        url_opener=_file_url_opener(archive),
    )

    # Track how many times the opener is called on the second install.
    call_count = {"value": 0}
    base_opener = _file_url_opener(archive)

    def counting_opener(url: str):
        call_count["value"] += 1
        return base_opener(url)

    report = install_split(
        bird_root=bird_root,
        split="dev",
        url=archive.as_uri(),
        url_opener=counting_opener,
        force=True,
    )
    assert call_count["value"] == 1
    assert report.download_bytes > 0


# ---------------------------------------------------------------------------
# install_split — failure modes
# ---------------------------------------------------------------------------


def test_install_split_rejects_non_https_url(tmp_path):
    """An ``http://`` URL is refused at the validate stage."""
    with pytest.raises(InstallError) as excinfo:
        install_split(
            bird_root=tmp_path / "bird",
            split="dev",
            url="http://insecure.example.com/dev.zip",
        )
    assert excinfo.value.stage == "validate"
    assert "https" in excinfo.value.message


def test_install_split_rejects_split_with_path_separator(tmp_path):
    """Path-traversal in the split name is rejected up front."""
    with pytest.raises(InstallError) as excinfo:
        install_split(
            bird_root=tmp_path / "bird",
            split="../evil",
        )
    assert excinfo.value.stage == "validate"


def test_install_split_rejects_truncated_download(tmp_path):
    """A response shorter than the advertised Content-Length is rejected."""
    archive = tmp_path / "src" / "dev.zip"
    archive.parent.mkdir(parents=True)
    _build_canonical_archive(archive, "dev")
    full_bytes = archive.read_bytes()

    def truncating_opener(url: str):
        from email.message import Message

        headers = Message()
        # Advertise the real size but only return half of it.
        headers["Content-Length"] = str(len(full_bytes))
        return urllib.request.addinfourl(  # type: ignore[arg-type]
            io.BytesIO(full_bytes[: len(full_bytes) // 2]),
            headers,
            url,
            200,
        )

    with pytest.raises(InstallError) as excinfo:
        install_split(
            bird_root=tmp_path / "bird",
            split="dev",
            url=archive.as_uri(),
            url_opener=truncating_opener,
        )
    assert excinfo.value.stage == "download"
    assert "truncated" in excinfo.value.message


def test_install_split_rejects_zip_slip(tmp_path):
    """An archive entry escaping the target dir is rejected at extract."""
    archive = tmp_path / "src" / "dev.zip"
    archive.parent.mkdir(parents=True)
    # Build an archive whose only canonical entries are valid, plus a
    # malicious entry whose name contains ``..`` so it would resolve
    # outside the staging directory.
    _build_canonical_archive(
        archive,
        "dev",
        extra_entries=[("../escape.txt", b"pwned")],
    )

    with pytest.raises(InstallError) as excinfo:
        install_split(
            bird_root=tmp_path / "bird",
            split="dev",
            url=archive.as_uri(),
            url_opener=_file_url_opener(archive),
        )
    assert excinfo.value.stage == "extract"


def test_install_split_rejects_archive_without_split_json(tmp_path):
    """An archive missing ``{split}.json`` fails at the normalize stage."""
    archive = tmp_path / "src" / "dev.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("README.txt", b"no JSON here")

    with pytest.raises(InstallError) as excinfo:
        install_split(
            bird_root=tmp_path / "bird",
            split="dev",
            url=archive.as_uri(),
            url_opener=_file_url_opener(archive),
        )
    assert excinfo.value.stage == "normalize"


def test_install_split_rejects_invalid_zip(tmp_path):
    """A response that isn't a ZIP at all fails at the download stage."""
    bogus = tmp_path / "src" / "dev.zip"
    bogus.parent.mkdir(parents=True)
    bogus.write_bytes(b"this is not a zip archive")

    def opener(url: str):
        from email.message import Message

        headers = Message()
        headers["Content-Length"] = str(bogus.stat().st_size)
        return urllib.request.addinfourl(  # type: ignore[arg-type]
            io.BytesIO(bogus.read_bytes()),
            headers,
            url,
            200,
        )

    with pytest.raises(InstallError) as excinfo:
        install_split(
            bird_root=tmp_path / "bird",
            split="dev",
            url=bogus.as_uri(),
            url_opener=opener,
        )
    assert excinfo.value.stage == "download"


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_cli_install_success(tmp_path, monkeypatch):
    """``bird-benchmark install`` returns 0 and prints a one-line summary."""
    archive = tmp_path / "src" / "dev.zip"
    archive.parent.mkdir(parents=True)
    _build_canonical_archive(archive, "dev")

    bird_root = tmp_path / "bird"

    # The CLI doesn't expose the ``url_opener`` test seam, so monkeypatch
    # the installer's default opener for the duration of this test.
    from bird_benchmark import installer as installer_mod

    monkeypatch.setattr(installer_mod, "_open_url", _file_url_opener(archive))

    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(
        [
            "install",
            "--bird-root",
            str(bird_root),
            "--split",
            "dev",
            "--url",
            archive.as_uri(),
        ],
        stderr=err,
        stdout=out,
    )
    assert code == EXIT_OK, err.getvalue()
    assert "installed split=dev" in out.getvalue()
    assert (bird_root / "dev" / "dev.json").is_file()


def test_cli_install_maps_install_error_to_config_exit(tmp_path):
    """An ``InstallError`` at validate maps to ``EXIT_CONFIG_ERROR``."""
    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(
        [
            "install",
            "--bird-root",
            str(tmp_path / "bird"),
            "--split",
            "dev",
            "--url",
            "http://insecure.example.com/dev.zip",
        ],
        stderr=err,
        stdout=out,
    )
    assert code == EXIT_CONFIG_ERROR
    # The error line should name the install stage so operators can grep
    # for ``[install:validate]`` in CI logs.
    assert "[install:validate]" in err.getvalue()


def test_cli_install_subparser_requires_bird_root_and_split():
    """Argparse refuses an install invocation missing required flags."""
    with pytest.raises(SystemExit):
        cli.main(["install"], stderr=io.StringIO(), stdout=io.StringIO())



# ---------------------------------------------------------------------------
# CLI ``list`` subcommand
# ---------------------------------------------------------------------------


def _install_synthetic_split(tmp_path: Path, split: str = "dev"):
    """Install a synthetic BIRD split and return the bird_root path.

    Reused across the ``list`` tests so each one starts from a real
    on-disk install (going through the installer end-to-end rather
    than hand-crafting the layout, so the tests exercise the same
    code path operators use in production).
    """

    archive = tmp_path / "src" / f"{split}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    _build_canonical_archive(archive, split)
    bird_root = tmp_path / "bird"
    install_split(
        bird_root=bird_root,
        split=split,
        url=archive.as_uri(),
        url_opener=_file_url_opener(archive),
    )
    return bird_root


def test_cli_list_default_format(tmp_path):
    """``list`` prints one ``id\\tdb_id\\tquestion`` line per Test_Case."""
    bird_root = _install_synthetic_split(tmp_path)

    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(
        [
            "list",
            "--bird-root",
            str(bird_root),
            "--split",
            "dev",
        ],
        stdout=out,
        stderr=err,
    )
    assert code == EXIT_OK
    lines = out.getvalue().splitlines()
    assert len(lines) == 1
    fields = lines[0].split("\t")
    assert fields[0] == "dev_0"
    assert fields[1] == "synth_db"
    assert "Synthetic question 0" in fields[2]


def test_cli_list_ids_only(tmp_path):
    """``--ids-only`` strips everything except the Test_Case_ID."""
    bird_root = _install_synthetic_split(tmp_path)

    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(
        [
            "list",
            "--bird-root",
            str(bird_root),
            "--split",
            "dev",
            "--ids-only",
        ],
        stdout=out,
        stderr=err,
    )
    assert code == EXIT_OK
    assert out.getvalue().strip() == "dev_0"


def test_cli_list_db_filter(tmp_path):
    """``--db`` filters to records whose ``db_id`` matches exactly."""
    bird_root = _install_synthetic_split(tmp_path)

    out = io.StringIO()
    err = io.StringIO()
    # The synthetic fixture has a single record under db_id ``synth_db``;
    # filtering by an unrelated db_id should produce no output.
    code = cli.main(
        [
            "list",
            "--bird-root",
            str(bird_root),
            "--split",
            "dev",
            "--db",
            "no_such_db",
        ],
        stdout=out,
        stderr=err,
    )
    assert code == EXIT_OK
    assert out.getvalue() == ""

    # Filtering by the real db_id yields the one record.
    out = io.StringIO()
    code = cli.main(
        [
            "list",
            "--bird-root",
            str(bird_root),
            "--split",
            "dev",
            "--db",
            "synth_db",
            "--ids-only",
        ],
        stdout=out,
        stderr=err,
    )
    assert code == EXIT_OK
    assert out.getvalue().strip() == "dev_0"


def test_cli_list_contains_filter_is_case_insensitive(tmp_path):
    """``--contains`` matches case-insensitively against the question."""
    bird_root = _install_synthetic_split(tmp_path)

    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(
        [
            "list",
            "--bird-root",
            str(bird_root),
            "--split",
            "dev",
            "--contains",
            "SYNTHETIC",
            "--ids-only",
        ],
        stdout=out,
        stderr=err,
    )
    assert code == EXIT_OK
    assert out.getvalue().strip() == "dev_0"


def test_cli_list_missing_install_returns_config_error(tmp_path):
    """Pointing ``list`` at a missing split surfaces ``EXIT_CONFIG_ERROR``."""
    out = io.StringIO()
    err = io.StringIO()
    code = cli.main(
        [
            "list",
            "--bird-root",
            str(tmp_path / "no_such_root"),
            "--split",
            "dev",
        ],
        stdout=out,
        stderr=err,
    )
    assert code == EXIT_CONFIG_ERROR
    assert "BIRD JSON file not found" in err.getvalue()
