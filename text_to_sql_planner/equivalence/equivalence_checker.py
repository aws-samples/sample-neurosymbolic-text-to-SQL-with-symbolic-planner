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

    # Build SMT-LIB script asserting the negation of equivalence
    script = _build_negated_equivalence_script(formula1, formula2)

    # Run two parallel cvc5 processes on the same script for robustness
    result = await _run_parallel_checks(script, config)
    return result


def _build_negated_equivalence_script(formula1: str, formula2: str) -> str:
    """Build SMT-LIB script asserting (not (= E1 E2)).

    - If unsat: the expressions are equivalent (no counterexample exists).
    - If sat: the expressions are not equivalent (counterexample found).
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
            # Extract the inner formula from (assert ...)
            assert1 = line[len("(assert "):-1]

    for line in lines2:
        if line.startswith("(declare-"):
            decls.add(line)
        elif line.startswith("(assert"):
            assert2 = line[len("(assert "):-1]

    result_lines = ["(set-logic ALL)"]
    result_lines.extend(sorted(decls))
    # Assert the negation of equivalence
    result_lines.append(f"(assert (not (= {assert1} {assert2})))")
    result_lines.append("(check-sat)")
    return "\n".join(result_lines)


async def _run_parallel_checks(
    script: str,
    config: EquivalenceCheckerConfig,
) -> EquivalenceResult:
    """Run two cvc5 processes in parallel on the same script.

    The script asserts (not (= E1 E2)):
    - unsat → equivalent (no counterexample exists)
    - sat → not equivalent (counterexample found)

    First conclusive result wins; the other is cancelled.
    """

    async def _run_cvc5() -> EquivalenceResult:
        """Run cvc5 with the script and interpret the result."""
        try:
            proc = await asyncio.create_subprocess_exec(
                config.cvc5_path,
                "--lang=smt2",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate(input=script.encode())
            output = stdout.decode().strip()
            returncode = proc.returncode or 0

            if returncode != 0:
                return IndeterminateResult(
                    reason=f"cvc5 exited with code {returncode}"
                )
            if output == "unsat":
                return EquivalentResult()
            elif output == "sat":
                return NotEquivalentResult()
            else:
                return IndeterminateResult(
                    reason=f"Unexpected cvc5 output: {output}"
                )
        except Exception as e:
            return IndeterminateResult(reason=str(e))

    # Create two parallel tasks for robustness (first to finish wins)
    task1 = asyncio.create_task(_run_cvc5())
    task2 = asyncio.create_task(_run_cvc5())

    try:
        done, pending = await asyncio.wait(
            {task1, task2},
            timeout=config.timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not done:
            # Timeout - cancel both
            task1.cancel()
            task2.cancel()
            return IndeterminateResult(reason="Timeout waiting for cvc5")

        # Get the first completed result
        for task in done:
            result = task.result()
            # Cancel remaining tasks
            for p in pending:
                p.cancel()
            return result

        return IndeterminateResult(reason="No result from cvc5")

    except asyncio.CancelledError:
        task1.cancel()
        task2.cancel()
        return IndeterminateResult(reason="Operation cancelled")
    except Exception as e:
        return IndeterminateResult(reason=str(e))
