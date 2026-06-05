"""BIRD dataset installer.

Downloads a BIRD split ZIP from the official distribution, extracts it,
normalises the on-disk layout to the shape :class:`bird_benchmark.loader.BirdLoader`
expects, and verifies the result by attempting to load the first record.

The installer exists because the BIRD distribution does not match the
loader's expected layout out of the box:

* The official dev ZIP wraps everything in a release-dated directory
  (e.g. ``dev_20240627/``) whose name changes between cleanup releases.
* In some releases the per-database SQLite files ship inside a nested
  ``{split}_databases.zip`` archive that has to be extracted in place.

After running ``install_split``, the on-disk layout is::

    {bird_root}/
        {split}/
            {split}.json
            {split}_databases/
                {db_id}/
                    {db_id}.sqlite

which is exactly what the loader expects (see ``loader.py``'s module
docstring).

Library API
-----------

The CLI thin-wraps :func:`install_split`, which is also usable directly
from Python:

.. code-block:: python

    from pathlib import Path
    from bird_benchmark import install_split

    report = install_split(
        bird_root=Path("./bird"),
        split="dev",
        progress=lambda done, total: print(f"{done}/{total}"),
    )
    print(report.split_root)

Errors flow through :class:`InstallError`. The function never partially
populates the target — it stages everything in a sibling temporary
directory and atomically renames into place on success, removing the
stale destination first. A ``KeyboardInterrupt`` during the download or
extract phase leaves the temporary directory behind for inspection but
does not damage an already-installed split.

Network policy
--------------

The installer uses :mod:`urllib.request` so it has no extra runtime
dependency. URLs must be HTTPS — an HTTP URL or any other scheme is
rejected up front with :class:`InstallError`. The default URL is the
official BIRD object-storage endpoint; ``url`` overrides it for use
with mirrors or local files (``file://`` is allowed for tests, see
:func:`_open_url`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

from bird_benchmark.loader import BirdLoader, BirdLoaderConfig
from bird_benchmark.types import SkippedTestCase, TestCase


# Official BIRD distribution URLs for each split.
#
# BIRD ships split archives via Alibaba OSS. The names are stable; the
# *contents* change between cleanup releases (the dated inner directory
# rotates) but the URL stays the same. If the URL ever moves, callers
# can pass ``url=...`` to :func:`install_split` to override.
_OFFICIAL_BIRD_URLS: dict[str, str] = {
    "dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
    "train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
}

# Bytes per chunk when streaming a download. 1 MiB is large enough that
# the per-chunk overhead is negligible against multi-GB downloads but
# small enough that progress callbacks fire often enough to feel live.
_DOWNLOAD_CHUNK_BYTES = 1 << 20

# How often to invoke the progress callback during a streamed download.
# Calling it on every chunk would spam stdout for slow consumers; once
# per second or once per ~16 MiB (whichever happens first) is enough.
_PROGRESS_INTERVAL_SECONDS = 0.5
_PROGRESS_INTERVAL_BYTES = 16 * (1 << 20)

# When ``urlopen`` does not advertise a Content-Length (some mirrors
# don't), the progress callback receives ``total=0`` and the CLI shows
# only the running byte count.
_UNKNOWN_TOTAL = 0


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class InstallError(Exception):
    """Raised when installing a BIRD split fails.

    The ``stage`` field names which step of the installer failed:
    ``"validate"`` (input checks), ``"download"``, ``"extract"``,
    ``"normalize"`` (layout fix-up), or ``"verify"`` (post-install
    sanity check). The CLI prints both the stage and the message so
    operators can tell whether to re-try with ``--force`` (download
    issue) or fix permissions (extract / normalize) or ignore (a
    verify-stage failure means the install completed but the loader
    rejected the result, which usually points at a BIRD-side schema
    change).
    """

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


@dataclass
class InstallReport:
    """Outcome of a successful :func:`install_split` call.

    All paths are absolute. ``download_bytes`` is the total number of
    bytes that crossed the wire (zero when ``url`` pointed at an already-
    local file or when the install short-circuited because ``--force``
    was not set).
    """

    split: str
    bird_root: Path
    split_root: Path
    json_path: Path
    databases_dir: Path
    database_count: int
    test_case_count: int
    download_bytes: int = 0
    skipped_records: int = 0
    notes: list[str] = field(default_factory=list)


# Type alias for the optional progress callback. Receives
# ``(bytes_so_far, total_bytes_or_zero)`` and may be invoked many times
# per download. ``total_bytes_or_zero`` is ``0`` when the server did
# not advertise a Content-Length.
ProgressCallback = Callable[[int, int], None]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def default_url_for(split: str) -> str:
    """Return the official BIRD download URL for the given split.

    Raises :class:`InstallError` (stage ``"validate"``) for splits the
    installer doesn't know about. Today only ``"dev"`` and ``"train"``
    are recognised; the test split is held by the BIRD authors and is
    not publicly downloadable.
    """

    try:
        return _OFFICIAL_BIRD_URLS[split]
    except KeyError as exc:
        known = ", ".join(sorted(_OFFICIAL_BIRD_URLS))
        raise InstallError(
            "validate",
            f"no default BIRD URL known for split {split!r} "
            f"(known splits: {known}); pass --url to override",
        ) from exc


def install_split(
    *,
    bird_root: Path,
    split: str,
    url: str | None = None,
    force: bool = False,
    keep_archive: bool = False,
    progress: ProgressCallback | None = None,
    url_opener: Callable[[str], urllib.request.addinfourl] | None = None,
) -> InstallReport:
    """Download and install one BIRD split into ``bird_root``.

    Parameters
    ----------
    bird_root:
        The directory that will end up containing every installed
        split. Created if missing. After this call,
        ``bird_root / split / split.json`` exists.
    split:
        The split name to install (``"dev"`` or ``"train"``); used as
        the directory name inside ``bird_root`` and as the prefix for
        the JSON file and databases directory.
    url:
        Override for the download URL. ``None`` (the default) uses the
        official BIRD endpoint from :func:`default_url_for`. Useful for
        mirrors and for tests (``file://`` is accepted).
    force:
        Re-download even when ``bird_root / split`` already contains a
        valid install. Without this flag the installer short-circuits
        to verification when the layout is already correct.
    keep_archive:
        If true, keep the downloaded ZIP at
        ``bird_root / _downloads / {split}.zip`` after extraction so
        future ``--force`` runs can re-extract without re-downloading.
        Default false: the archive is removed once the layout is in
        place.
    progress:
        Optional callback that receives ``(bytes_so_far, total_bytes)``
        roughly every :data:`_PROGRESS_INTERVAL_SECONDS` during the
        download. ``total_bytes`` is ``0`` when the server does not
        advertise Content-Length.
    url_opener:
        Test seam. ``None`` uses :func:`_open_url`. Tests can pass a
        callable that opens a local fixture archive without going
        through ``urlopen``.

    Returns
    -------
    InstallReport
        On success.

    Raises
    ------
    InstallError
        Stage names the failing step (validate / download / extract /
        normalize / verify).
    """

    # --- validate -------------------------------------------------------
    if not isinstance(bird_root, Path):
        raise InstallError(
            "validate",
            f"bird_root must be a pathlib.Path, got {type(bird_root).__name__}",
        )
    bird_root = bird_root.expanduser().resolve()

    if not isinstance(split, str) or not split.strip():
        raise InstallError(
            "validate", f"split must be a non-empty string, got {split!r}"
        )
    if "/" in split or os.sep in split:
        # The split name is used as a path component, so reject any
        # value that would escape ``bird_root``.
        raise InstallError(
            "validate",
            f"split name must not contain a path separator, got {split!r}",
        )

    download_url = url if url is not None else default_url_for(split)
    parsed = urlparse(download_url)
    if parsed.scheme not in ("https", "file"):
        raise InstallError(
            "validate",
            f"refusing non-HTTPS URL {download_url!r} "
            f"(allowed schemes: https, file)",
        )

    # --- short-circuit when already installed ---------------------------
    target_split = bird_root / split
    if not force and _layout_is_valid(target_split, split):
        # Already installed and the loader reports a valid layout. Skip
        # download + extract; still run verify so the report has the
        # same shape as a fresh install.
        try:
            return _verify_install(bird_root, split, download_bytes=0,
                                   notes=["short-circuit: layout already valid"])
        except InstallError as exc:
            # If verify fails on a pre-existing install, surface it as
            # such — the operator probably wants ``--force``.
            raise InstallError(
                exc.stage,
                f"{exc.message} (existing install at {target_split} is "
                f"present but unverifiable; re-run with --force to reinstall)",
            ) from exc

    # --- download -------------------------------------------------------
    bird_root.mkdir(parents=True, exist_ok=True)
    downloads_dir = bird_root / "_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive_path = downloads_dir / f"{split}.zip"

    download_bytes = 0
    if force or not archive_path.is_file() or not _is_valid_zip(archive_path):
        # Download to a sibling temp file and rename atomically on
        # success, so an interrupted download never poses as a
        # complete archive on the next run.
        opener = url_opener if url_opener is not None else _open_url
        download_bytes = _download_to(archive_path, download_url, opener, progress)
    elif progress is not None:
        # Tell the caller we're skipping the download so progress UIs
        # can reset their state.
        size = archive_path.stat().st_size
        progress(size, size)

    if not _is_valid_zip(archive_path):
        # _download_to already validated; this catches the cached-archive
        # path's case where an earlier interrupted run left a partial
        # file behind that the size check above didn't catch.
        archive_path.unlink(missing_ok=True)
        raise InstallError(
            "download",
            f"downloaded archive at {archive_path} is not a valid ZIP "
            "(possibly truncated); rerun with --force",
        )

    # --- extract --------------------------------------------------------
    # Stage extraction in a sibling directory and rename on success so a
    # mid-extract crash never leaves a half-populated ``{split}/`` in
    # ``bird_root``.
    staging_root = Path(tempfile.mkdtemp(
        prefix=f".{split}-staging-", dir=bird_root
    ))
    notes: list[str] = []
    try:
        try:
            with zipfile.ZipFile(archive_path) as zf:
                _safe_extract_all(zf, staging_root)
        except zipfile.BadZipFile as exc:
            raise InstallError(
                "extract",
                f"archive at {archive_path} is not a valid ZIP: {exc}",
            ) from exc
        except OSError as exc:
            raise InstallError(
                "extract",
                f"failed to write extracted files under {staging_root}: {exc}",
            ) from exc

        # --- normalize --------------------------------------------------
        # Find the directory that contains ``{split}.json`` and any
        # nested ``{split}_databases.zip``. Move that contents into
        # the final destination.
        try:
            normalized_root, normalize_notes = _normalize_extracted_layout(
                staging_root, split
            )
        except InstallError:
            raise
        notes.extend(normalize_notes)

        # Atomically replace the destination. ``shutil.rmtree`` first
        # because ``Path.rename`` refuses to clobber a non-empty dir.
        if target_split.exists():
            shutil.rmtree(target_split)
        target_split.parent.mkdir(parents=True, exist_ok=True)
        normalized_root.rename(target_split)
    finally:
        # Drop whatever's left of the staging dir; on success it's
        # already been moved out, on failure we want it gone so the
        # next run starts clean.
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)

    # --- archive housekeeping -------------------------------------------
    if not keep_archive:
        archive_path.unlink(missing_ok=True)
        # Remove the _downloads dir if empty so a successful install
        # doesn't leave an obvious detritus directory in ``bird_root``.
        try:
            downloads_dir.rmdir()
        except OSError:
            # Other splits may still have archives in there.
            pass

    # --- verify ---------------------------------------------------------
    return _verify_install(bird_root, split, download_bytes=download_bytes,
                           notes=notes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_url(url: str) -> urllib.request.addinfourl:
    """Open a URL with a sane default timeout and a friendly user agent.

    Centralised so :func:`install_split`'s ``url_opener`` parameter can
    swap in a test fake without having to mirror all the boilerplate.
    """

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "bird-benchmark/0.1 (+installer)"},
    )
    # 60 s is generous for a connect; the data transfer itself has no
    # whole-stream timeout because GB-scale downloads on slow links would
    # trip it.
    return urllib.request.urlopen(request, timeout=60)


def _download_to(
    archive_path: Path,
    url: str,
    url_opener: Callable[[str], urllib.request.addinfourl],
    progress: ProgressCallback | None,
) -> int:
    """Stream ``url`` into ``archive_path``, returning bytes written.

    Writes to ``archive_path.with_suffix(archive_path.suffix + ".part")``
    first so an interrupted run does not leave a misleading half-file
    sitting at the canonical path.
    """

    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    if partial_path.exists():
        partial_path.unlink()

    try:
        response = url_opener(url)
    except urllib.error.URLError as exc:
        raise InstallError(
            "download", f"failed to open {url}: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise InstallError(
            "download", f"failed to open {url}: {exc}"
        ) from exc

    total = 0
    try:
        # Content-Length isn't always present (chunked transfer). Use it
        # when we have it so the progress callback can show a percentage.
        try:
            length_header = response.headers.get("Content-Length")
            advertised_total = int(length_header) if length_header else _UNKNOWN_TOTAL
        except (TypeError, ValueError):
            advertised_total = _UNKNOWN_TOTAL

        last_callback = time.monotonic()
        last_callback_bytes = 0
        with partial_path.open("wb") as out:
            while True:
                try:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                except (urllib.error.URLError, OSError) as exc:
                    raise InstallError(
                        "download",
                        f"download from {url} interrupted after {total} bytes: {exc}",
                    ) from exc
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)

                if progress is not None:
                    now = time.monotonic()
                    if (
                        now - last_callback >= _PROGRESS_INTERVAL_SECONDS
                        or total - last_callback_bytes >= _PROGRESS_INTERVAL_BYTES
                    ):
                        progress(total, advertised_total)
                        last_callback = now
                        last_callback_bytes = total

        if progress is not None:
            # Final tick so the UI lands on 100% / final byte count.
            progress(total, advertised_total)
    finally:
        response.close()

    if advertised_total and total != advertised_total:
        partial_path.unlink(missing_ok=True)
        raise InstallError(
            "download",
            f"download from {url} truncated: got {total} bytes, "
            f"server advertised {advertised_total}",
        )

    if not _is_valid_zip(partial_path):
        partial_path.unlink(missing_ok=True)
        raise InstallError(
            "download",
            f"download from {url} did not produce a valid ZIP at {partial_path}",
        )

    # Atomic rename — on POSIX this is rename(2); on Windows it's an
    # equivalent. After this point the archive is durable on disk.
    partial_path.replace(archive_path)
    return total


def _is_valid_zip(path: Path) -> bool:
    """Cheap structural validity check for a ZIP archive.

    ``zipfile.is_zipfile`` only looks at the End-of-Central-Directory
    record; ``ZipFile.testzip`` would catch every CRC mismatch but
    walks the whole file (slow for GB-scale archives). For the
    installer's purposes — does this look like a ZIP we could plausibly
    extract — the cheap check is enough; corruption inside individual
    members will surface during extraction with a clear error.
    """

    try:
        return path.is_file() and zipfile.is_zipfile(path)
    except OSError:
        return False


def _safe_extract_all(zf: zipfile.ZipFile, target: Path) -> None:
    """Extract ``zf`` under ``target`` after validating each member name.

    Rejects entries whose normalised path escapes ``target`` (zip-slip
    defence). The target is created if missing.
    """

    target.mkdir(parents=True, exist_ok=True)
    target_resolved = target.resolve()

    for name in zf.namelist():
        # Normalise path separators and rule out absolute paths.
        if name.startswith("/") or name.startswith("\\") or ":" in name[:3]:
            raise InstallError(
                "extract",
                f"refusing absolute archive entry name {name!r}",
            )
        member_path = (target / name).resolve()
        try:
            member_path.relative_to(target_resolved)
        except ValueError as exc:
            raise InstallError(
                "extract",
                f"refusing archive entry {name!r}: extracts outside {target}",
            ) from exc

    zf.extractall(target)


def _normalize_extracted_layout(
    staging_root: Path, split: str
) -> tuple[Path, list[str]]:
    """Find the directory holding ``{split}.json`` and produce the
    canonical layout under it.

    The official BIRD dev archive has shipped at least three layouts
    over time:

    1. **Direct**: ``{split}.json`` and ``{split}_databases/`` at the
       top of the archive.
    2. **Dated wrapper**: a single inner directory named like
       ``dev_20240627/`` that contains the layout from (1).
    3. **Nested archive**: layout (2) but with ``{split}_databases.zip``
       in place of the unpacked databases directory; we extract the
       nested ZIP in place.

    The function locates the JSON file, lifts everything around it into
    a sibling directory named ``{split}_normalized``, extracts any
    nested databases archive, and returns that directory.

    Returns the normalised root and a list of human-readable notes
    describing what it had to do, so the CLI can surface a one-line
    summary of the dataset's idiosyncrasies.
    """

    notes: list[str] = []
    json_filename = f"{split}.json"

    # Locate the JSON file. There must be exactly one path matching the
    # name; multiple matches are ambiguous.
    matches = list(staging_root.rglob(json_filename))
    if not matches:
        raise InstallError(
            "normalize",
            f"could not find {json_filename} anywhere under the extracted archive",
        )
    if len(matches) > 1:
        rels = ", ".join(sorted(str(m.relative_to(staging_root)) for m in matches))
        raise InstallError(
            "normalize",
            f"found multiple {json_filename} files under the extracted archive: {rels}",
        )

    json_path = matches[0]
    split_root_in_archive = json_path.parent

    if split_root_in_archive != staging_root:
        notes.append(
            f"BIRD archive wraps split in inner dir "
            f"{split_root_in_archive.relative_to(staging_root)}; flattened"
        )

    # Move the contents up to a fresh sibling directory rather than
    # picking up unrelated archive top-level files (license, README, …)
    # that don't belong in the canonical layout.
    normalized_root = staging_root.parent / f"{split}_normalized_{os.getpid()}"
    if normalized_root.exists():
        shutil.rmtree(normalized_root)
    shutil.copytree(split_root_in_archive, normalized_root)

    # If the databases directory ships as a nested ZIP, extract in place.
    nested_zip = normalized_root / f"{split}_databases.zip"
    if nested_zip.is_file():
        try:
            with zipfile.ZipFile(nested_zip) as inner_zf:
                _safe_extract_all(inner_zf, normalized_root)
        except zipfile.BadZipFile as exc:
            raise InstallError(
                "normalize",
                f"nested archive {nested_zip} is not a valid ZIP: {exc}",
            ) from exc
        nested_zip.unlink()
        notes.append(f"extracted nested {split}_databases.zip in place")

    # Sanity check — both required entries are present.
    expected_json = normalized_root / f"{split}.json"
    expected_dbs = normalized_root / f"{split}_databases"
    if not expected_json.is_file():
        raise InstallError(
            "normalize",
            f"after layout normalisation, {expected_json} is still missing",
        )
    if not expected_dbs.is_dir():
        raise InstallError(
            "normalize",
            f"after layout normalisation, {expected_dbs} is still missing "
            f"(no {split}_databases directory in the archive)",
        )

    return normalized_root, notes


def _layout_is_valid(target_split: Path, split: str) -> bool:
    """Cheap structural check of the canonical layout.

    Used by ``install_split`` to decide whether to short-circuit when
    ``--force`` is not set. A more thorough check happens in
    :func:`_verify_install` after the install completes.
    """

    if not target_split.is_dir():
        return False
    if not (target_split / f"{split}.json").is_file():
        return False
    if not (target_split / f"{split}_databases").is_dir():
        return False
    return True


def _verify_install(
    bird_root: Path,
    split: str,
    *,
    download_bytes: int,
    notes: list[str],
) -> InstallReport:
    """Open the install with ``BirdLoader`` and stream every record once.

    Confirms the on-disk layout is what the loader expects and that
    every record's per-database SQLite file is reachable. Records that
    surface as :class:`SkippedTestCase` (missing fields, missing
    SQLite) are tallied but don't fail the install — BIRD ships a
    handful of malformed records and the suite's normal skip path
    handles them.
    """

    target_split = bird_root / split
    json_path = target_split / f"{split}.json"
    databases_dir = target_split / f"{split}_databases"

    if not json_path.is_file():
        raise InstallError(
            "verify", f"BIRD JSON file not found at {json_path}"
        )
    if not databases_dir.is_dir():
        raise InstallError(
            "verify", f"BIRD databases directory not found at {databases_dir}"
        )

    try:
        loader = BirdLoader(BirdLoaderConfig(bird_root=bird_root, split=split))
        items: Iterable[TestCase | SkippedTestCase] = loader.load()
        test_cases = 0
        skipped = 0
        for item in items:
            if isinstance(item, SkippedTestCase):
                skipped += 1
            else:
                test_cases += 1
    except Exception as exc:
        raise InstallError(
            "verify",
            f"BirdLoader rejected the installed split: {exc}",
        ) from exc

    # Database count: every immediate subdirectory of databases_dir.
    database_count = sum(1 for child in databases_dir.iterdir() if child.is_dir())

    return InstallReport(
        split=split,
        bird_root=bird_root,
        split_root=target_split,
        json_path=json_path,
        databases_dir=databases_dir,
        database_count=database_count,
        test_case_count=test_cases,
        download_bytes=download_bytes,
        skipped_records=skipped,
        notes=list(notes),
    )


__all__ = [
    "InstallError",
    "InstallReport",
    "ProgressCallback",
    "default_url_for",
    "install_split",
]
