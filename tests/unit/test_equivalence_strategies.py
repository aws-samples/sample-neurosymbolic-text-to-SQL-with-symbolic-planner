"""Tests for the multi-strategy cvc5 driver in
:mod:`text_to_sql_planner.equivalence.equivalence_checker`.

We stub :func:`asyncio.create_subprocess_exec` so each "cvc5 process"
returns a pre-baked verdict for the strategy it was invoked with. This
lets us exercise:

- the default strategy list has at least three entries;
- a decisive result (``sat`` or ``unsat``) wins immediately and
  cancels the other strategies;
- an ``unknown`` from one strategy does NOT defeat a still-pending
  decisive answer from another;
- if every strategy returns ``unknown`` (or a process error), the
  driver returns the last :class:`IndeterminateResult`;
- the wall-clock timeout applies across the whole strategy set, not
  per-strategy.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

import text_to_sql_planner.equivalence.equivalence_checker as eqmod
from text_to_sql_planner.equivalence.equivalence_checker import (
    EquivalenceCheckerConfig,
    EquivalentResult,
    IndeterminateResult,
    NotEquivalentResult,
    _run_parallel_checks,
)


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _StubProc:
    """Mimics the small subset of ``asyncio.subprocess.Process`` the
    runner uses: ``communicate(input=...)`` returns ``(stdout, stderr)``
    after an optional delay, and ``returncode`` is set on the way out.
    """

    def __init__(self, stdout: bytes, returncode: int = 0, delay: float = 0.0):
        self._stdout = stdout
        self.returncode = returncode
        self._delay = delay

    async def communicate(self, input: bytes | None = None):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._stdout, b""


def _stub_cvc5(verdict_per_strategy: dict[str, _StubProc]):
    """Return a coroutine factory that stubs ``create_subprocess_exec``.

    ``verdict_per_strategy`` maps a key like ``"--mbqi"`` (or ``""`` for
    the default strategy with no extra flag beyond ``--lang=smt2``) to
    the :class:`_StubProc` that should be returned when cvc5 is
    invoked with that flag.
    """

    async def _fake(cmd, *args, **kwargs):
        # ``args`` includes everything after ``cvc5_path``: we look for
        # the strategy-discriminating flag.
        key = ""
        for arg in args:
            if arg in verdict_per_strategy:
                key = arg
                break
        proc = verdict_per_strategy.get(key, _StubProc(b"unknown", 0))
        return proc

    return _fake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_strategy_list_has_three_entries():
    """The default config exposes three different cvc5 strategies."""
    cfg = EquivalenceCheckerConfig()
    assert len(cfg.strategies) >= 3
    flags = [" ".join(s) for s in cfg.strategies]
    # Default: bare ``--lang=smt2`` (no quantifier-specific flag).
    assert any("--lang=smt2" in s and "--mbqi" not in s and "--full-saturate-quant" not in s for s in flags)
    # MBQI strategy.
    assert any("--mbqi" in s for s in flags)
    # Full-saturation strategy.
    assert any("--full-saturate-quant" in s for s in flags)


@pytest.mark.asyncio
async def test_first_decisive_wins(monkeypatch: pytest.MonkeyPatch):
    """The fastest decisive verdict is returned, cancelling the others."""
    # Default strategy: returns ``sat`` immediately.
    # MBQI strategy: would return ``unknown`` after a delay.
    # Full-saturate: would also delay.
    stubs = {
        "": _StubProc(b"sat", 0, delay=0.0),
        "--mbqi": _StubProc(b"unknown", 0, delay=0.5),
        "--full-saturate-quant": _StubProc(b"unknown", 0, delay=0.5),
    }
    monkeypatch.setattr(eqmod.asyncio, "create_subprocess_exec", _stub_cvc5(stubs))

    cfg = EquivalenceCheckerConfig(timeout_seconds=2.0)
    result = await _run_parallel_checks("(check-sat)", cfg)

    assert isinstance(result, NotEquivalentResult)


@pytest.mark.asyncio
async def test_unknown_does_not_beat_pending_decisive(monkeypatch: pytest.MonkeyPatch):
    """When the fastest strategy returns ``unknown``, the driver waits
    for the slower-but-decisive strategy instead of giving up.

    Default strategy: ``unknown`` after a tiny delay.
    MBQI strategy: ``unsat`` after a slightly longer delay.
    """
    stubs = {
        "": _StubProc(b"unknown", 0, delay=0.05),
        "--mbqi": _StubProc(b"unsat", 0, delay=0.15),
        "--full-saturate-quant": _StubProc(b"unknown", 0, delay=0.5),
    }
    monkeypatch.setattr(eqmod.asyncio, "create_subprocess_exec", _stub_cvc5(stubs))

    cfg = EquivalenceCheckerConfig(timeout_seconds=2.0)
    result = await _run_parallel_checks("(check-sat)", cfg)

    assert isinstance(result, EquivalentResult)


@pytest.mark.asyncio
async def test_all_unknown_returns_indeterminate(monkeypatch: pytest.MonkeyPatch):
    """When every strategy returns ``unknown`` the driver gives up
    with :class:`IndeterminateResult`."""
    stubs = {
        "": _StubProc(b"unknown", 0, delay=0.01),
        "--mbqi": _StubProc(b"unknown", 0, delay=0.02),
        "--full-saturate-quant": _StubProc(b"unknown", 0, delay=0.03),
    }
    monkeypatch.setattr(eqmod.asyncio, "create_subprocess_exec", _stub_cvc5(stubs))

    cfg = EquivalenceCheckerConfig(timeout_seconds=2.0)
    result = await _run_parallel_checks("(check-sat)", cfg)

    assert isinstance(result, IndeterminateResult)


@pytest.mark.asyncio
async def test_global_timeout_is_shared_across_strategies(monkeypatch: pytest.MonkeyPatch):
    """The wall-clock budget applies across the whole strategy set —
    granting each strategy its own ``timeout_seconds`` would silently
    triple the worst-case latency."""
    # All three strategies hang past the budget. The driver must give
    # up *once* at ``timeout_seconds``, not three times.
    stubs = {
        "": _StubProc(b"unsat", 0, delay=10.0),
        "--mbqi": _StubProc(b"unsat", 0, delay=10.0),
        "--full-saturate-quant": _StubProc(b"unsat", 0, delay=10.0),
    }
    monkeypatch.setattr(eqmod.asyncio, "create_subprocess_exec", _stub_cvc5(stubs))

    cfg = EquivalenceCheckerConfig(timeout_seconds=0.1)
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    result = await _run_parallel_checks("(check-sat)", cfg)
    elapsed = loop.time() - t0

    assert isinstance(result, IndeterminateResult)
    assert "Timeout" in result.reason or "timeout" in result.reason.lower()
    # The whole call should finish well before 1.0 second — proves we
    # didn't retry/serialise the strategies.
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_process_error_treated_as_indeterminate(monkeypatch: pytest.MonkeyPatch):
    """A non-zero exit code is treated as Indeterminate (so the other
    strategies still get a chance to win)."""
    stubs = {
        "": _StubProc(b"", 1, delay=0.01),       # default errors
        "--mbqi": _StubProc(b"unsat", 0, delay=0.05),  # MBQI succeeds
        "--full-saturate-quant": _StubProc(b"unknown", 0, delay=0.5),
    }
    monkeypatch.setattr(eqmod.asyncio, "create_subprocess_exec", _stub_cvc5(stubs))

    cfg = EquivalenceCheckerConfig(timeout_seconds=2.0)
    result = await _run_parallel_checks("(check-sat)", cfg)

    assert isinstance(result, EquivalentResult)
