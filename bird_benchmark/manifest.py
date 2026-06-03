"""Append-only JSONL Run_Manifest for the BIRD benchmark framework.

The manifest is the suite driver's crash-resilience contract: every
``Run_Result`` is written as a single JSON object on its own line,
followed by ``flush()`` + ``os.fsync()`` so a SIGKILL after the call
cannot leave the file in a partial-record state. A truncated last line
(no trailing ``\\n``) is detected at read time and dropped, so the
``--resume`` driver re-runs the Test_Case that was in flight when the
process died.

See Requirements 8.2, 8.3, and 8.5.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from bird_benchmark.types import RunResult


class ManifestParseError(Exception):
    """Raised when reading a Run_Manifest produces a malformed line.

    Carries the manifest path, the 1-indexed line number of the offending
    record, and the underlying parse error message so the CLI can surface
    it verbatim per Req 8.5.
    """

    def __init__(self, path: Path, line_number: int, parse_error: str):
        self.path = path
        self.line_number = line_number
        self.parse_error = parse_error
        super().__init__(
            f"Manifest at {path} has malformed JSON on line {line_number}: "
            f"{parse_error}"
        )


def _default_serializer(obj: Any) -> Any:
    """Fallback ``json.dumps`` serialiser for non-standard types.

    ``RunResult`` carries ``Path`` and ``Verdict`` values that the standard
    encoder cannot handle directly. ``Verdict`` is a ``str``-Enum so
    ``json.dumps`` already treats it as a string in most paths, but we keep
    the explicit fallback so any future enum field keeps round-tripping
    through its ``.value``.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Cannot serialise {type(obj).__name__}: {obj!r}")


def _serialize_run_result(result: RunResult) -> dict[str, Any]:
    """Convert a ``RunResult`` to a JSON-serialisable dict.

    ``dataclasses.asdict`` recurses through nested dataclasses; ``Verdict``
    fields stay as ``Verdict`` instances (which are also ``str`` subclasses
    so the JSON encoder accepts them).
    """
    return asdict(result)


class Manifest:
    """Append-only JSONL Run_Manifest reader/writer.

    Open the file with :meth:`open_for_append`, call :meth:`append` once
    per ``Run_Result``, and call :meth:`read_completed_ids` at the start of
    a ``--resume`` run to learn which Test_Case_IDs have already finished.
    """

    def __init__(self, path: Path):
        self.path = path
        self._file = None

    @classmethod
    def open_for_append(cls, path: Path) -> "Manifest":
        """Open the manifest in append mode, creating the parent dir if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        m = cls(path)
        m._file = open(path, "a", encoding="utf-8")
        return m

    def append(self, result: RunResult) -> None:
        """Append one ``RunResult`` as a single JSONL line.

        Writes ``json.dumps(...) + "\\n"`` in one ``write`` call, then
        ``flush`` + ``os.fsync`` so a SIGKILL right after returning leaves
        the on-disk manifest with a complete record (Req 8.2).
        """
        if self._file is None:
            raise RuntimeError(
                "Manifest is not open for append; call Manifest.open_for_append first"
            )
        line = (
            json.dumps(
                _serialize_run_result(result),
                default=_default_serializer,
            )
            + "\n"
        )
        self._file.write(line)
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "Manifest":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def read_completed_ids(self) -> set[str]:
        """Stream the manifest and return the set of completed Test_Case_IDs.

        A trailing record without a closing ``\\n`` is treated as truncated
        (the writer was killed mid-flush) and dropped; every prior complete
        line is kept, so the suite driver re-runs the truncated Test_Case
        on ``--resume`` (Req 8.3).

        Raises :class:`ManifestParseError` for any complete line that is
        not valid JSON (Req 8.5).
        """
        if not self.path.exists():
            return set()

        completed: set[str] = set()
        with self.path.open(encoding="utf-8") as f:
            text = f.read()

        if not text:
            return completed

        # ``str.split('\n')`` always returns at least one element. If the
        # file ends with ``\n`` the last element is ``""`` — that's the
        # well-formed trailer we drop. If the file does NOT end with
        # ``\n`` the last element is the truncated line — we drop that too
        # so the suite driver re-runs the Test_Case that was in flight.
        lines = text.split("\n")
        complete_lines = lines[:-1]

        for i, raw in enumerate(complete_lines, start=1):
            if not raw.strip():
                # Skip blank lines defensively. The writer never produces
                # them, so silently tolerating them costs us nothing.
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ManifestParseError(self.path, i, str(e)) from e
            tid = obj.get("test_case_id")
            if tid:
                completed.add(tid)
        return completed

    def read_all(self) -> list[dict[str, Any]]:
        """Return every parsed Run_Result as a dict, with the same
        truncated-tail and malformed-line semantics as
        :meth:`read_completed_ids`.

        Useful for the reporter, which needs the full Run_Result body
        rather than just the Test_Case_IDs.
        """
        if not self.path.exists():
            return []

        results: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as f:
            text = f.read()

        if not text:
            return results

        lines = text.split("\n")
        complete_lines = lines[:-1]

        for i, raw in enumerate(complete_lines, start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ManifestParseError(self.path, i, str(e)) from e
            results.append(obj)
        return results
