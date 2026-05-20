"""Equivalence checking between DRC expressions using cvc5."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Union

from text_to_sql_planner.types.drc import DRCExpression

from .smt_converter import convert_to_smt


def _indent_smt(script: str, width: int = 80) -> str:
    """Pretty-print an SMT-LIB script with indentation.

    Short lines stay as-is. Long parenthesized expressions get broken
    across multiple lines with indentation reflecting nesting depth.
    """
    output_lines: list[str] = []
    for line in script.split("\n"):
        if len(line) <= width:
            output_lines.append(line)
        else:
            output_lines.append(_indent_sexp(line, width))
    return "\n[cvc5]   ".join(output_lines)


def _indent_sexp(text: str, width: int = 80) -> str:
    """Indent a single long S-expression across multiple lines."""
    result: list[str] = []
    indent = 0
    i = 0
    current_line: list[str] = []
    current_len = 0

    while i < len(text):
        ch = text[i]

        if ch == '(':
            token = _read_token_at(text, i)
            # Check if the whole sub-expression from here fits on one line
            end = _find_matching_paren(text, i)
            sub_expr = text[i:end + 1] if end != -1 else text[i:]

            if current_len + len(sub_expr) <= width - indent * 2:
                # Fits on current line
                current_line.append(sub_expr)
                current_len += len(sub_expr) + 1
                i = end + 1 if end != -1 else len(text)
            else:
                # Doesn't fit — open paren on this line, indent contents
                # Read the opening "(" and the operator/keyword after it
                head = _read_head(text, i)
                if current_line:
                    result.append(" " * (indent * 2) + " ".join(current_line))
                    current_line = []
                    current_len = 0
                result.append(" " * (indent * 2) + head)
                indent += 1
                i += len(head)
                current_line = []
                current_len = 0
        elif ch == ')':
            if current_line:
                result.append(" " * (indent * 2) + " ".join(current_line) + ")")
                current_line = []
                current_len = 0
            else:
                # Close paren on its own or appended to last line
                if result:
                    result[-1] = result[-1] + ")"
                else:
                    result.append(")")
            indent = max(0, indent - 1)
            i += 1
        elif ch == ' ':
            i += 1
        elif ch == '"':
            # Read string literal
            end_q = text.index('"', i + 1) if '"' in text[i + 1:] else len(text) - 1
            token = text[i:end_q + 1]
            current_line.append(token)
            current_len += len(token) + 1
            i = end_q + 1
        else:
            # Read atom
            end_a = i
            while end_a < len(text) and text[end_a] not in ' ()':
                end_a += 1
            token = text[i:end_a]
            current_line.append(token)
            current_len += len(token) + 1
            i = end_a

    if current_line:
        result.append(" " * (indent * 2) + " ".join(current_line))

    return "\n[cvc5]   ".join(result)


def _find_matching_paren(text: str, start: int) -> int:
    """Find the index of the matching closing paren for the open paren at `start`."""
    depth = 0
    i = start
    in_string = False
    while i < len(text):
        ch = text[i]
        if ch == '"' and not in_string:
            in_string = True
        elif ch == '"' and in_string:
            in_string = False
        elif not in_string:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _read_head(text: str, start: int) -> str:
    """Read '(' plus the first atom/keyword after it (e.g. '(assert ', '(exists ')."""
    i = start + 1  # skip '('
    # skip whitespace
    while i < len(text) and text[i] == ' ':
        i += 1
    # read the keyword
    end = i
    while end < len(text) and text[end] not in ' ()':
        end += 1
    return text[start:end + 1] if end < len(text) and text[end] == ' ' else text[start:end]


def _read_token_at(text: str, i: int) -> str:
    """Read a full token starting at position i."""
    if text[i] == '(':
        end = _find_matching_paren(text, i)
        return text[i:end + 1] if end != -1 else text[i:]
    end = i
    while end < len(text) and text[end] not in ' ()':
        end += 1
    return text[i:end]


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
    """Check if two DRC expressions are logically equivalent using cvc5."""
    if config is None:
        config = EquivalenceCheckerConfig()

    # Early exit: if result variable counts differ, expressions can't be equivalent
    if len(expr1.result_variables) != len(expr2.result_variables):
        print(f"\n> ⚡ **cvc5:** Arity mismatch ({len(expr1.result_variables)} vs {len(expr2.result_variables)}) → `not_equivalent` (skipped cvc5)\n")
        return NotEquivalentResult()

    print(f"\n### cvc5 equivalence check\n", flush=True)

    # Convert conditions to SMT-LIB formulas
    formula1 = convert_to_smt(expr1.condition)
    formula2 = convert_to_smt(expr2.condition)

    # Build SMT-LIB script asserting the negation of equivalence
    script = _build_negated_equivalence_script(formula1, formula2)

    print(f"#### SMT-LIB script ({len(script)} chars)\n")
    print(f"```smt2\n{script}\n```\n")
    print(f"Invoking `{config.cvc5_path}` (timeout={config.timeout_seconds}s)...\n", flush=True)

    # Run two parallel cvc5 processes on the same script for robustness
    result = await _run_parallel_checks(script, config)

    status_icon = "✅" if result.status == "equivalent" else "❌" if result.status == "not_equivalent" else "⚠️"
    reason_str = f" — {result.reason}" if hasattr(result, 'reason') and result.reason else ""
    print(f"**Result:** {status_icon} `{result.status}`{reason_str}\n")

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
            assert1 = line[len("(assert "):-1]

    for line in lines2:
        if line.startswith("(declare-"):
            decls.add(line)
        elif line.startswith("(assert"):
            assert2 = line[len("(assert "):-1]

    result_lines = ["(set-logic ALL)"]
    result_lines.extend(sorted(decls))
    result_lines.append(f"(assert (not (= {assert1} {assert2})))")
    result_lines.append("(check-sat)")
    return "\n".join(result_lines)


async def _run_parallel_checks(
    script: str,
    config: EquivalenceCheckerConfig,
) -> EquivalenceResult:
    """Run two cvc5 processes in parallel on the same script."""

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
            stdout, stderr = await proc.communicate(input=script.encode())
            output = stdout.decode().strip()
            err_output = stderr.decode().strip()
            returncode = proc.returncode or 0

            if err_output:
                print(f"[cvc5]   stderr: {err_output[:200]}")

            if returncode != 0:
                print(f"[cvc5]   Process exited with code {returncode}")
                return IndeterminateResult(
                    reason=f"cvc5 exited with code {returncode}: {err_output[:100]}"
                )

            print(f"[cvc5]   Output: {output}")

            if output == "unsat":
                return EquivalentResult()
            elif output == "sat":
                return NotEquivalentResult()
            else:
                return IndeterminateResult(
                    reason=f"Unexpected cvc5 output: {output[:100]}"
                )
        except FileNotFoundError:
            return IndeterminateResult(reason=f"cvc5 binary not found at '{config.cvc5_path}'")
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
            task1.cancel()
            task2.cancel()
            print(f"[cvc5]   TIMEOUT after {config.timeout_seconds}s")
            return IndeterminateResult(reason="Timeout waiting for cvc5")

        # Get the first completed result
        for task in done:
            result = task.result()
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
