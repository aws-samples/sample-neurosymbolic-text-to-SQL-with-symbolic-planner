"""Expected_Fail_List loader and stale-entry detection.

The Expected_Fail_List is a user-supplied file naming Test_Cases whose gold
SQL is known to be over-specified, ambiguous, or otherwise outside the
framework's intended evaluation scope. The runner uses the list as a one-way
mask: a non-equivalent underlying verdict for a listed Test_Case becomes
``Verdict.expected_fail`` in the report, while an equivalent verdict surfaces
a stale-entry warning so operators can prune the list (Req 12.3 / 12.4 /
12.6).

File format (Req 12.1):

* one Test_Case_ID per line
* leading and trailing whitespace on each line is stripped
* blank lines and lines starting with ``#`` are ignored

A ``None`` path (no ``--expected-fail`` supplied on the CLI) returns an
empty set so the rest of the pipeline can treat the override uniformly. A
non-``None`` path that cannot be opened or read raises
:class:`ExpectedFailLoadError` so the CLI can exit non-zero with a clear
message naming the file (Req 12.2).
"""

from __future__ import annotations

from pathlib import Path


class ExpectedFailLoadError(Exception):
    """Raised when the Expected_Fail_List file cannot be opened or read.

    The CLI surfaces both ``path`` and ``reason`` so the operator can fix
    the underlying file before re-running.
    """

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot load Expected_Fail_List at {path}: {reason}")


def load(path: Path | None) -> set[str]:
    """Load the Expected_Fail_List from ``path``.

    Returns an empty set when ``path`` is ``None`` so callers do not need
    to special-case the no-list configuration. When ``path`` is supplied,
    each non-blank, non-comment line (after stripping whitespace) is added
    to the returned set as a Test_Case_ID.

    Raises:
        ExpectedFailLoadError: ``path`` is non-``None`` but the file is
            missing, unreadable, or otherwise cannot be opened.
    """

    if path is None:
        return set()

    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise ExpectedFailLoadError(path, str(exc)) from exc

    expected: set[str] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        expected.add(line)
    return expected


def detect_stale(expected: set[str], seen: set[str]) -> set[str]:
    """Return Test_Case_IDs in ``expected`` that did not appear in ``seen``.

    Used by the suite driver to surface Expected_Fail_List entries that no
    longer correspond to any Test_Case in the requested split, so operators
    can prune the list (Req 12.6). Trivial set difference; named so the call
    site reads as intent rather than arithmetic.
    """

    return expected - seen
