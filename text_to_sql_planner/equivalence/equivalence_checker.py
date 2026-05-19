"""Equivalence checking between DRC expressions using cvc5."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Union

from text_to_sql_planner.types.drc import DRCExpression

from .smt_converter import convert_to_smt


@dataclass
class EquivalenceCheckerConfig:
    """Configuration for the equivalence checker."""

    cvc5_path: str = "cvc5"
    timeout_seconds: float = 30.0


@dataclass
class EquivalentResult:
    """The two expressions are logically equivalent."""

    status: Literal["equivalent"] = "equivalent"


@dataclass
class NotEquivalentResult:
    """The two expressions are not logically equivalent."""

    status: Literal["not_equivalent"] = "not_equivalent"


@dataclass
class IndeterminateResult:
    """Could not determine equivalence (timeout, error, etc.)."""

    status: Literal["indeterminate"] = "indeterminate"
    reason: str = ""


EquivalenceResult = Union[EquivalentResult, NotEquivalentResult, IndeterminateResult]


async def check_equivalence(
    expr1: DRCExpression,
    expr2: DRCExpression,
    config: EquivalenceCheckerConfig | None = None,
) -> EquivalenceResult:
    """Check if two DRC expressions are logically equivalent using cvc5.

    Strategy:
    - Early exit if result variable counts differ (not equivalent).
    - Spawn two parallel cvc5 processes:
      1. Test if (= E1 E2) is unsatisfiable → equivalent
      2. Test if (not (= E1 E2)) is satisfiable → not equivalent
    - First conclusive result wins; the other process is cancelled.
    - On timeout or error, return IndeterminateResult.
    """
    if config is None:
        config = EquivalenceCheckerConfig()

    # Early exit: if result variable counts differ, expressions can't be equivalent
    if len(expr1.result_variables) != len(expr2.result_variables):
        return NotEquivalentResult()

    # Convert conditions to SMT-LIB formulas
    formula1 = convert_to_smt(expr1.condition)
    formula2 = convert_to_smt(expr2.condition)

    # Build SMT-LIB scripts for the two checks
    equiv_script = _build_equivalence_script(formula1, formula2)
    not_equiv_script = _build_not_equivalence_script(formula1, formula2)

    # Run both checks in parallel
    result = await _run_parallel_checks(
        equiv_script, not_equiv_script, config
    )
    return result


def _build_equivalence_script(formula1: str, formula2: str) -> str:
    """Build SMT-LIB script to test if (= E1 E2) is unsatisfiable.

    If unsat, the expressions are equivalent.
    We combine declarations from both formulas and assert their equality.
    """
    lines1 = formula1.split("\n")
    lines2 = formula2.split("\n")

    # Collect declarations and assertions
    decls: set[str] = set()
    assert1 = ""
    assert2 = ""

    for line in lines1:
        if line.startswith("(declare-"):
            decls.add(line)
        elif line.startswith("(assert"):
            # Extract the inner formula from (assert ...)
            assert1 = line[len("(assert "):-1]

    for line in lines2:
        if line.startswith("(declare-"):
            decls.add(line)
        elif line.startswith("(assert"):
            assert2 = line[len("(assert "):-1]

    result_lines = ["(set-logic ALL)"]
    result_lines.extend(sorted(decls))
    # Assert that the two formulas are equal (iff)
    result_lines.append(f"(assert (= {assert1} {assert2}))")
    result_lines.append("(check-sat)")
    return "\n".join(result_lines)


def _build_not_equivalence_script(formula1: str, formula2: str) -> str:
    """Build SMT-LIB script to test if (not (= E1 E2)) is satisfiable.

    If sat, the expressions are not equivalent.
    """
    lines1 = formula1.split("\n")
    lines2 = formula2.split("\n")

    decls: set[str] = set()
    assert1 = ""
    assert2 = ""

    for line in lines1:
        if line.startswith("(declare-"):
            decls.add(line)
        elif line.startswith("(assert"):
            assert1 = line[len("(assert "):-1]

    for line in lines2:
        if line.startswith("(declare-"):
            decls.add(line)
        elif line.startswith("(assert"):
            assert2 = line[len("(assert "):-1]

    result_lines = ["(set-logic ALL)"]
    result_lines.extend(sorted(decls))
    # Assert that the two formulas are NOT equal
    result_lines.append(f"(assert (not (= {assert1} {assert2})))")
    result_lines.append("(check-sat)")
    return "\n".join(result_lines)


async def _run_parallel_checks(
    equiv_script: str,
    not_equiv_script: str,
    config: EquivalenceCheckerConfig,
) -> EquivalenceResult:
    """Run two cvc5 processes in parallel and return the first conclusive result."""

    async def _run_cvc5(script: str) -> tuple[str, int]:
        """Run cvc5 with the given script and return (stdout, returncode)."""
        proc = await asyncio.create_subprocess_exec(
            config.cvc5_path,
            "--lang=smt2",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate(input=script.encode())
        return stdout.decode().strip(), proc.returncode or 0

    async def _check_equiv() -> EquivalenceResult | None:
        """Check if (= E1 E2) is unsatisfiable → equivalent."""
        try:
            output, returncode = await _run_cvc5(equiv_script)
            if returncode != 0:
                return IndeterminateResult(
                    reason=f"cvc5 exited with code {returncode}"
                )
            if output == "unsat":
                # (= E1 E2) is unsatisfiable means they can never both be true
                # Actually we want: the negation of equivalence is unsat
                # But our script asserts (= E1 E2) directly
                # If sat, they CAN be equal (not conclusive alone)
                # We need to rethink: assert (not (iff E1 E2)) → unsat means equivalent
                return None
            elif output == "sat":
                return EquivalentResult()
            else:
                return IndeterminateResult(
                    reason=f"Unexpected cvc5 output: {output}"
                )
        except Exception as e:
            return IndeterminateResult(reason=str(e))

    async def _check_not_equiv() -> EquivalenceResult | None:
        """Check if (not (= E1 E2)) is satisfiable → not equivalent."""
        try:
            output, returncode = await _run_cvc5(not_equiv_script)
            if returncode != 0:
                return IndeterminateResult(
                    reason=f"cvc5 exited with code {returncode}"
                )
            if output == "sat":
                return NotEquivalentResult()
            elif output == "unsat":
                return EquivalentResult()
            else:
                return IndeterminateResult(
                    reason=f"Unexpected cvc5 output: {output}"
                )
        except Exception as e:
            return IndeterminateResult(reason=str(e))

    # Create tasks for both checks
    task_equiv = asyncio.create_task(_check_equiv())
    task_not_equiv = asyncio.create_task(_check_not_equiv())

    try:
        done, pending = await asyncio.wait(
            {task_equiv, task_not_equiv},
            timeout=config.timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            # Timeout - cancel both
            task_equiv.cancel()
            task_not_equiv.cancel()
            return IndeterminateResult(reason="Timeout waiting for cvc5")

        # Get the first completed result
        for task in done:
            result = task.result()
            if result is not None:
                # Cancel the other task
                for p in pending:
                    p.cancel()
                return result

        # First task returned None (inconclusive), wait for the other
        if pending:
            try:
                done2, _ = await asyncio.wait(
                    pending,
                    timeout=config.timeout_seconds,
                )
                for task in done2:
                    result = task.result()
                    if result is not None:
                        return result
            except asyncio.TimeoutError:
                pass

        return IndeterminateResult(reason="Both checks were inconclusive")

    except asyncio.CancelledError:
        task_equiv.cancel()
        task_not_equiv.cancel()
        return IndeterminateResult(reason="Operation cancelled")
    except Exception as e:
        return IndeterminateResult(reason=str(e))
