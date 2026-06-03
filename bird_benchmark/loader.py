"""BIRD dataset loader.

Reads a local BIRD download (a directory containing per-split JSON metadata
files plus per-database SQLite files) and yields :class:`TestCase` records
ready for the runner.

The on-disk layout this loader assumes (the layout BIRD distributes):

    {bird_root}/
        {split}/
            {split}.json
            {split}_databases/
                {db_id}/
                    {db_id}.sqlite

A missing JSON or SQLite file fails the whole load with
:class:`BirdLoadError`; a per-record problem (missing required field, missing
per-record SQLite file) yields a :class:`SkippedTestCase` so the suite driver
can record the skip and keep going. See Requirement 1 in
``requirements.md``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from bird_benchmark.types import SkippedTestCase, TestCase


@dataclass
class BirdLoaderConfig:
    """Where to read the BIRD download from.

    ``bird_root`` is the directory that contains the per-split subfolders;
    ``split`` is the BIRD split name (``"dev"``, ``"test"``, etc.) and is
    used as both a directory name and a JSON filename prefix.
    """

    bird_root: Path
    split: str


class BirdLoadError(Exception):
    """Raised when the BIRD download is missing or malformed at the file level.

    The CLI surfaces both ``path`` and ``reason`` so the operator can fix
    the underlying file before re-running.
    """

    def __init__(self, path: str | Path, reason: str) -> None:
        self.path = str(path)
        self.reason = reason
        super().__init__(f"{self.path}: {reason}")


class BirdLoader:
    """Stream BIRD records as :class:`TestCase` / :class:`SkippedTestCase`."""

    def __init__(self, config: BirdLoaderConfig) -> None:
        self._config = config

    # --- path helpers -------------------------------------------------------

    def _json_path(self) -> Path:
        split = self._config.split
        return self._config.bird_root / split / f"{split}.json"

    def _sqlite_path(self, db_id: str) -> Path:
        split = self._config.split
        return (
            self._config.bird_root
            / split
            / f"{split}_databases"
            / db_id
            / f"{db_id}.sqlite"
        )

    # --- public API ---------------------------------------------------------

    def load(self) -> Iterator[TestCase | SkippedTestCase]:
        """Yield one item per record in the BIRD JSON file, in source order.

        Raises :class:`BirdLoadError` for missing JSON or unparsable JSON.
        Per-record problems (missing required field, missing SQLite file)
        come back as :class:`SkippedTestCase` instead of raising so the
        suite driver can keep going (Req 1.7).
        """

        json_path = self._json_path()
        if not json_path.is_file():
            raise BirdLoadError(json_path, "BIRD JSON file not found")

        try:
            with json_path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        except json.JSONDecodeError as exc:
            raise BirdLoadError(
                json_path, f"failed to parse BIRD JSON: {exc.msg}"
            ) from exc

        if not isinstance(records, list):
            raise BirdLoadError(
                json_path,
                f"expected a JSON array of records, got {type(records).__name__}",
            )

        # Cache schema strings per db_id so we open each SQLite file at most
        # once per load, even when many records share a database.
        schema_cache: dict[str, str] = {}

        for index, record in enumerate(records):
            yield from self._handle_record(index, record, schema_cache)

    # --- per-record handling ------------------------------------------------

    def _handle_record(
        self,
        index: int,
        record: object,
        schema_cache: dict[str, str],
    ) -> Iterable[TestCase | SkippedTestCase]:
        split = self._config.split

        if not isinstance(record, dict):
            yield SkippedTestCase(
                test_case_id=f"{split}_record_{index}",
                reason=(
                    f"expected JSON object record, got {type(record).__name__}"
                ),
            )
            return

        question_id = record.get("question_id")
        if question_id is None or (
            isinstance(question_id, str) and question_id.strip() == ""
        ):
            test_case_id = f"{split}_record_{index}"
        else:
            test_case_id = f"{split}_{question_id}"

        # Required fields: db_id, question, SQL (note: capitalised SQL).
        for field_name in ("db_id", "question", "SQL"):
            if not _is_non_empty_string(record.get(field_name)):
                yield SkippedTestCase(
                    test_case_id=test_case_id,
                    reason=f"missing field '{field_name}'",
                )
                return

        db_id: str = record["db_id"]
        question: str = record["question"]
        gold_sql: str = record["SQL"]

        evidence_raw = record.get("evidence")
        if isinstance(evidence_raw, str) and evidence_raw.strip() != "":
            evidence = evidence_raw
        else:
            evidence = ""

        sqlite_path = self._sqlite_path(db_id)
        if not sqlite_path.is_file():
            yield SkippedTestCase(
                test_case_id=test_case_id,
                reason=f"sqlite database not found at {sqlite_path}",
            )
            return

        if db_id not in schema_cache:
            schema_cache[db_id] = _read_schema(sqlite_path)

        yield TestCase(
            test_case_id=test_case_id,
            split=split,
            db_id=db_id,
            schema=schema_cache[db_id],
            question=question,
            evidence=evidence,
            gold_sql=gold_sql,
        )


def _is_non_empty_string(value: object) -> bool:
    """Required-field check: present, a string, and non-empty after strip."""

    return isinstance(value, str) and value.strip() != ""


def _read_schema(sqlite_path: Path) -> str:
    """Return the joined ``CREATE TABLE`` statements for the SQLite database.

    Joins the ``sql`` column of ``sqlite_master`` for every table with
    ``;\\n`` (Req 1.4). The ``ORDER BY name`` clause makes the output
    deterministic across runs.
    """

    with sqlite3.connect(str(sqlite_path)) as connection:
        cursor = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL "
            "ORDER BY name"
        )
        statements = [row[0] for row in cursor.fetchall()]

    return ";\n".join(statements)
