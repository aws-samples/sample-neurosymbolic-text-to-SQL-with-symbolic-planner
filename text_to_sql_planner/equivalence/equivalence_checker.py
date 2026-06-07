"""Equivalence checking between DRC expressions using cvc5."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal, Union

from text_to_sql_planner.types.drc import DRCExpression

from .smt_converter import convert_to_smt


def _indent_smt(script: str, width: int = 80) -> str:
    """Pretty-print an SMT-LIB script with indentation.

    Each top-level command stays on its own line. Long S-expressions
    get broken across multiple lines with 2-space indentation per nesting level.
    """
    output_lines: list[str] = []
    for line in script.split("\n"):
        if len(line) <= width:
            output_lines.append(line)
        else:
            output_lines.append(_indent_sexp(line, width))
    return "\n".join(output_lines)


def _indent_sexp(text: str, width: int = 80) -> str:
    """Indent a single long S-expression across multiple lines.

    Uses 2-space indentation per nesting level. Sub-expressions that
    fit within the remaining width stay on one line.
    """
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

    return "\n".join(result)


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
    """Configuration for the equivalence checker.

    ``strategies`` is a list of cvc5 argument profiles to run in
    parallel. Each profile is a list of CLI flags (without the binary
    name and without the SMT-LIB script — those are added by the
    runner). The first profile that returns a *decisive* verdict
    (``sat`` or ``unsat``) wins; ``unknown`` and process errors are
    treated as non-decisive and the driver waits for the others.

    Default strategies cover three different quantifier-instantiation
    tactics:

    1. **Default E-matching** — fast on most formulas, weak on
       deep quantifier alternation.
    2. **Model-based quantifier instantiation (``--mbqi``)** — slower
       on easy formulas but reliably handles ∀∃ alternation,
       which is exactly the regime that defeats E-matching for
       counting-style queries (``≥N`` vs ``exactly-N``).
    3. **Full saturation (``--full-saturate-quant``)** — falls back
       to exhaustive instantiation when the heuristics give up.
       Slow but decisive on harder cases.

    Running them in parallel costs one extra cvc5 process per check
    (still cheap relative to the LLM calls in the planner loop) and
    lifts most ``unknown`` outcomes to ``unsat``/``sat``.
    """

    cvc5_path: str = "cvc5"
    timeout_seconds: float = 30.0
    strategies: list[list[str]] = field(
        default_factory=lambda: [
            ["--lang=smt2"],
            ["--lang=smt2", "--mbqi"],
            ["--lang=smt2", "--full-saturate-quant"],
        ]
    )


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
    schema_types: dict[str, str] | None = None,
    axioms: list[str] | None = None,
    label: str | None = None,
    lhs_label: str | None = None,
    rhs_label: str | None = None,
) -> EquivalenceResult:
    """Check if two DRC expressions are logically equivalent using cvc5.

    Two DRC expressions are equivalent iff:
    1. They have the same result variable structure (same types, same aggregates)
    2. For all values of the result variables, their conditions are equivalent (C1 ↔ C2)

    Args:
        schema_types: Optional dict mapping column_name -> "Int"|"String" from the DB schema.
        axioms: Optional list of SMT-LIB assertion strings (each one a complete
            ``(assert ...)`` form) to prepend to the equivalence script. The
            equivalence proof then proceeds *under* these axioms — useful for
            referential-integrity facts that the schema declares but the DRC
            condition doesn't restate (e.g. every row in a child table has a
            matching row in the parent). Each axiom must use the same predicate
            symbols as the DRC sides, with the same per-position sort signature.
        label: Optional human-readable description of *what* is being
            compared, appended to the ``### cvc5 equivalence check`` banner
            in the run log. The same function is called from two distinct
            contexts ("planner relation vs target DRC" during planning and
            "generated DRC vs gold DRC" during BIRD evaluation) and they
            looked identical in the markdown trace; the label
            disambiguates them. Defaults to no label (the bare banner) so
            existing call sites are unaffected.
        lhs_label / rhs_label: Optional short tags for the two DRC sides.
            When supplied, ``Result:`` lines append a hint of the form
            ``(LHS = generated, RHS = gold)`` and the SMT-LIB script gets
            comment headers identifying each side. Useful when reading a
            ``not_equivalent`` script after the fact: knowing which side
            is which makes the difference immediately legible.
    """
    if config is None:
        config = EquivalenceCheckerConfig()

    from text_to_sql_planner.types.drc import ColumnVariable, AggregateVariable

    # Helper: render the optional ``(LHS = generated, RHS = gold)``
    # disambiguator. Empty string when neither side label was supplied
    # so the existing log format stays clean.
    sides_hint = ""
    if lhs_label and rhs_label:
        sides_hint = f" (LHS = {lhs_label}, RHS = {rhs_label})"

    # Early exit: if result variable counts differ, expressions can't be equivalent
    if len(expr1.result_variables) != len(expr2.result_variables):
        print(
            f"\n> ⚡ **cvc5:** Arity mismatch "
            f"({len(expr1.result_variables)} vs {len(expr2.result_variables)})"
            f"{sides_hint} → `not_equivalent` (skipped cvc5)\n"
        )
        return NotEquivalentResult()

    # Early exit: result variable structure must match (column vs aggregate, function names)
    for rv1, rv2 in zip(expr1.result_variables, expr2.result_variables):
        if type(rv1) != type(rv2):
            print(
                f"\n> ⚡ **cvc5:** Result variable type mismatch "
                f"({type(rv1).__name__} vs {type(rv2).__name__})"
                f"{sides_hint} → `not_equivalent` (skipped cvc5)\n"
            )
            return NotEquivalentResult()
        if isinstance(rv1, AggregateVariable) and isinstance(rv2, AggregateVariable):
            if rv1.function != rv2.function:
                print(
                    f"\n> ⚡ **cvc5:** Aggregate function mismatch "
                    f"({rv1.function} vs {rv2.function})"
                    f"{sides_hint} → `not_equivalent` (skipped cvc5)\n"
                )
                return NotEquivalentResult()

    banner = "### cvc5 equivalence check"
    if label:
        banner = f"{banner}: {label}"
    print(f"\n{banner}\n", flush=True)

    # Build the proper equivalence check script. When side labels were
    # supplied, prepend each side's assertion in the script with a
    # comment line so the reader can tell at a glance which formula is
    # which when investigating ``not_equivalent`` results.
    script = _build_equivalence_script(
        expr1, expr2,
        schema_types=schema_types,
        axioms=axioms,
        lhs_label=lhs_label,
        rhs_label=rhs_label,
    )

    print(f"#### SMT-LIB script ({len(script)} chars)\n")
    print(f"```smt2\n{_indent_smt(script)}\n```\n")
    print(f"Invoking `{config.cvc5_path}` (timeout={config.timeout_seconds}s)...\n", flush=True)

    # Run two parallel cvc5 processes on the same script for robustness
    result = await _run_parallel_checks(script, config)

    status_icon = "✅" if result.status == "equivalent" else "❌" if result.status == "not_equivalent" else "⚠️"
    reason_str = f" — {result.reason}" if hasattr(result, 'reason') and result.reason else ""
    print(f"**Result:** {status_icon} `{result.status}`{sides_hint}{reason_str}\n")

    return result


def _build_equivalence_script(
    expr1: DRCExpression,
    expr2: DRCExpression,
    schema_types: dict[str, str] | None = None,
    axioms: list[str] | None = None,
    lhs_label: str | None = None,
    rhs_label: str | None = None,
) -> str:
    """Build SMT-LIB script to check equivalence of two DRC expressions.

    For ``{x1,...,xn | C1}`` and ``{y1,...,yn | C2}`` we check::

        (assert (not (forall ((r1 Sort) ... (rn Sort)) (= C1' C2'))))
        (check-sat)

    where ``r1..rn`` are *fresh* shared names, ``C1'`` is ``C1`` with each
    result variable substituted to its shared name, and ``C2'`` is the
    same for ``C2``. The substitution is essential: the two DRC
    expressions may use different names for the same logical result
    column (e.g. one side carries an operator-introduced ``_1`` suffix
    while the other uses the bare column name). Without substitution,
    ``C2`` would reference declared free constants instead of the
    universally quantified result variables, and the equivalence check
    would be vacuous.

    If unsat: the expressions define the same relation (equivalent).
    If sat: there exists a tuple where they differ (not equivalent).
    """
    from text_to_sql_planner.types.drc import ColumnVariable, AggregateVariable
    from .smt_converter import collect_symbols

    # Per-side names of the result variables (in positional order).
    def _rv_name(rv) -> str:
        if isinstance(rv, ColumnVariable):
            return rv.name
        if isinstance(rv, AggregateVariable):
            return rv.column
        raise ValueError(f"Unknown result variable type: {type(rv).__name__}")

    rv_names_1 = [_rv_name(rv) for rv in expr1.result_variables]
    rv_names_2 = [_rv_name(rv) for rv in expr2.result_variables]

    # ------------------------------------------------------------------
    # Pre-SMT preprocessing: trivial-equality elimination + unused-slot
    # pruning. The two conditions are processed *jointly* in pass 2 so
    # that the predicate signature stays consistent across both sides.
    # Result-variable names on either side are explicitly kept alive so
    # the slot pruner doesn't drop them from membership terms.
    # ------------------------------------------------------------------
    from .smt_preprocessing import preprocess_for_smt_pair

    keep_names = set(rv_names_1) | set(rv_names_2)
    pre1, pre2 = preprocess_for_smt_pair(
        [expr1.condition, expr2.condition],
        keep_names=keep_names,
    )
    expr1 = type(expr1)(result_variables=expr1.result_variables, condition=pre1)
    expr2 = type(expr2)(result_variables=expr2.result_variables, condition=pre2)

    # Collect symbols and infer types from both sides.
    rels1, vars1 = collect_symbols(expr1.condition)
    rels2, vars2 = collect_symbols(expr2.condition)

    from .smt_converter import collect_var_types, _collect_relation_sorts, _convert_node
    var_types1 = collect_var_types(expr1.condition)
    var_types2 = collect_var_types(expr2.condition)
    all_var_types: dict[str, str] = {}
    for vt in (var_types1, var_types2):
        for v, t in vt.items():
            if t == "String" or v not in all_var_types:
                all_var_types[v] = t

    if schema_types:
        for v, t in schema_types.items():
            if t == "String":
                all_var_types[v] = "String"

    all_relations: dict[str, int] = {}
    for name, arity in rels1.items():
        all_relations[name] = max(all_relations.get(name, 0), arity)
    for name, arity in rels2.items():
        all_relations[name] = max(all_relations.get(name, 0), arity)

    rel_sorts: dict[str, list[str]] = {}
    _collect_relation_sorts(expr1.condition, all_var_types, rel_sorts)
    _collect_relation_sorts(expr2.condition, all_var_types, rel_sorts)
    _propagate_types_from_relations(expr1.condition, rel_sorts, all_var_types)
    _propagate_types_from_relations(expr2.condition, rel_sorts, all_var_types)
    rel_sorts = {}
    _collect_relation_sorts(expr1.condition, all_var_types, rel_sorts)
    _collect_relation_sorts(expr2.condition, all_var_types, rel_sorts)

    # Pick fresh shared names for the universally quantified result
    # variables. Avoid collisions with anything declared or referenced
    # anywhere in either expression.
    reserved = set(vars1) | set(vars2) | set(rv_names_1) | set(rv_names_2)
    shared_names: list[str] = []
    for i in range(len(rv_names_1)):
        base = f"_rv_{i}"
        candidate = base
        counter = 2
        while candidate in reserved:
            candidate = f"{base}_{counter}"
            counter += 1
        shared_names.append(candidate)
        reserved.add(candidate)

    # Per-side scope maps so each formula's result-variable references
    # resolve to the shared names.
    scope1 = {orig: shared for orig, shared in zip(rv_names_1, shared_names)}
    scope2 = {orig: shared for orig, shared in zip(rv_names_2, shared_names)}

    # Determine the type for each shared name (String wins over Int).
    def _shared_sort(idx: int) -> str:
        n1 = rv_names_1[idx]
        n2 = rv_names_2[idx]
        t1 = all_var_types.get(n1, "Int")
        t2 = all_var_types.get(n2, "Int")
        if t1 == "String" or t2 == "String":
            return "String"
        return "Int"

    # Convert each condition under its substitution scope.
    formula1 = _convert_node(expr1.condition, all_var_types, scope1)
    formula2 = _convert_node(expr2.condition, all_var_types, scope2)

    # Build the script.
    lines: list[str] = []
    lines.append("(set-logic ALL)")

    for rel_name, arity in sorted(all_relations.items()):
        if rel_name in rel_sorts:
            sorts = " ".join(rel_sorts[rel_name])
        else:
            sorts = " ".join(["Int"] * arity)
        lines.append(f"(declare-fun {rel_name} ({sorts}) Bool)")

    # Free variables = everything used in either condition that wasn't a
    # result variable on its own side. With substitution in place, the
    # result variables vanish from the formula, replaced by the shared
    # names, so they should NOT be declared as free constants.
    bound_originals = set(rv_names_1) | set(rv_names_2)
    all_variables = vars1 | vars2
    for var in sorted(all_variables - bound_originals):
        sort = all_var_types.get(var, "Int")
        lines.append(f"(declare-const {var} {sort})")

    if shared_names:
        bindings = " ".join(
            f"({name} {_shared_sort(i)})" for i, name in enumerate(shared_names)
        )
        equivalence_assert = (
            f"(assert (not (forall ({bindings}) (= {formula1} {formula2}))))"
        )
    else:
        equivalence_assert = f"(assert (not (= {formula1} {formula2})))"

    # Inject any caller-supplied axioms BEFORE the negated equivalence
    # assertion. Standard SMT-LIB semantics: the solver looks for a
    # model that satisfies every assertion, so axioms here become
    # facts the equivalence proof gets to assume. The runner uses
    # this for foreign-key referential-integrity axioms.
    if axioms:
        for axiom in axioms:
            lines.append(axiom)

    # When the caller named the two sides, drop a comment above the
    # equivalence assertion that maps "first formula" -> LHS and
    # "second formula" -> RHS. SMT-LIB semicolon comments are
    # stripped by cvc5; humans investigating a ``not_equivalent``
    # script see them.
    if lhs_label and rhs_label:
        lines.append(
            f";; (= LHS RHS)  --  LHS = {lhs_label}, RHS = {rhs_label}"
        )
    lines.append(equivalence_assert)

    lines.append("(check-sat)")
    return "\n".join(lines)


async def _run_parallel_checks(
    script: str,
    config: EquivalenceCheckerConfig,
) -> EquivalenceResult:
    """Run cvc5 in parallel under several different quantifier-
    instantiation strategies; the first *decisive* answer wins.

    A "decisive" answer is ``sat`` (NotEquivalent) or ``unsat``
    (Equivalent). ``unknown`` and process errors are not decisive —
    when one of the strategies returns one of those, we keep waiting
    for the others. We stop only when:

    - some strategy returns a decisive answer (cancel the rest, win),
    - or all strategies have completed without a decisive answer
      (return Indeterminate with the most informative reason),
    - or the wall-clock budget is exhausted (return Indeterminate
      "timeout").

    Running multiple strategies costs one extra cvc5 process per
    check, but it lifts a large fraction of ``unknown`` outcomes —
    in particular the "≥N vs exactly-N" pattern that defeats default
    E-matching but is solved easily by ``--mbqi``.
    """
    strategies = list(config.strategies) or [["--lang=smt2"]]
    tasks = [
        asyncio.create_task(_run_cvc5_with_strategy(script, config, args))
        for args in strategies
    ]

    decisive_result: EquivalenceResult | None = None
    last_indeterminate: IndeterminateResult | None = None

    loop = asyncio.get_event_loop()
    deadline = loop.time() + config.timeout_seconds

    try:
        # Wait for tasks to complete one at a time; stop early on the
        # first decisive result, but keep waiting on indeterminates.
        # The wall-clock budget is shared across all strategies — we
        # don't grant each strategy a fresh ``timeout_seconds`` window.
        pending = set(tasks)
        while pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                for p in pending:
                    p.cancel()
                print(f"[cvc5]   TIMEOUT after {config.timeout_seconds}s")
                return IndeterminateResult(
                    reason=f"Timeout waiting for cvc5 after {config.timeout_seconds}s"
                )
            done, pending = await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                # Wall-clock budget exhausted with no completions.
                for p in pending:
                    p.cancel()
                print(f"[cvc5]   TIMEOUT after {config.timeout_seconds}s")
                return IndeterminateResult(
                    reason=f"Timeout waiting for cvc5 after {config.timeout_seconds}s"
                )

            for task in done:
                result = task.result()
                if isinstance(result, (EquivalentResult, NotEquivalentResult)):
                    decisive_result = result
                    break
                elif isinstance(result, IndeterminateResult):
                    last_indeterminate = result

            if decisive_result is not None:
                # Cancel any still-running strategies and return.
                for p in pending:
                    p.cancel()
                return decisive_result

        # All strategies completed without a decisive answer.
        if last_indeterminate is not None:
            return last_indeterminate
        return IndeterminateResult(reason="No decisive cvc5 result from any strategy")

    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        return IndeterminateResult(reason="Operation cancelled")
    except Exception as e:
        for t in tasks:
            t.cancel()
        return IndeterminateResult(reason=str(e))


async def _run_cvc5_with_strategy(
    script: str,
    config: EquivalenceCheckerConfig,
    args: list[str],
) -> EquivalenceResult:
    """Run cvc5 once with the given CLI arg list and interpret the result."""
    try:
        proc = await asyncio.create_subprocess_exec(
            config.cvc5_path,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=script.encode())
        output = stdout.decode(errors="replace").strip()
        err_output = stderr.decode(errors="replace").strip()
        returncode = proc.returncode or 0

        if returncode != 0:
            return IndeterminateResult(
                reason=(
                    f"cvc5({' '.join(args)}) exited with code {returncode}: "
                    f"{err_output[:100]}"
                )
            )

        if output == "unsat":
            return EquivalentResult()
        if output == "sat":
            return NotEquivalentResult()
        return IndeterminateResult(
            reason=f"cvc5({' '.join(args)}) returned {output[:100]}"
        )
    except FileNotFoundError:
        return IndeterminateResult(
            reason=f"cvc5 binary not found at '{config.cvc5_path}'"
        )
    except asyncio.CancelledError:
        # Bubble up cancellation cleanly so the driver's bookkeeping
        # works.
        raise
    except Exception as e:
        return IndeterminateResult(reason=str(e))


def _propagate_types_from_relations(condition, rel_sorts: dict[str, list[str]], var_types: dict[str, str]) -> None:
    """Propagate types from relation sort signatures to variables.

    If a relation is declared as (Int Int String) and a membership uses
    variables (a, b, c) at those positions, then c must be String.
    """
    from text_to_sql_planner.types.drc import (
        MembershipNode, LogicalConnectiveNode, NotNode,
        QuantifierNode, ComparisonNode, ArithmeticNode, FunctionCallNode,
    )

    if condition is None:
        return

    if isinstance(condition, MembershipNode):
        if condition.relation in rel_sorts:
            sorts = rel_sorts[condition.relation]
            for i, var in enumerate(condition.variables):
                if i < len(sorts) and sorts[i] == "String":
                    var_types[var] = "String"

    elif isinstance(condition, LogicalConnectiveNode):
        _propagate_types_from_relations(condition.left, rel_sorts, var_types)
        _propagate_types_from_relations(condition.right, rel_sorts, var_types)

    elif isinstance(condition, NotNode):
        _propagate_types_from_relations(condition.operand, rel_sorts, var_types)

    elif isinstance(condition, QuantifierNode):
        _propagate_types_from_relations(condition.body, rel_sorts, var_types)

    elif isinstance(condition, ComparisonNode):
        _propagate_types_from_relations(condition.left, rel_sorts, var_types)
        _propagate_types_from_relations(condition.right, rel_sorts, var_types)

    elif isinstance(condition, ArithmeticNode):
        _propagate_types_from_relations(condition.left, rel_sorts, var_types)
        _propagate_types_from_relations(condition.right, rel_sorts, var_types)

    elif isinstance(condition, FunctionCallNode):
        for arg in condition.arguments:
            _propagate_types_from_relations(arg, rel_sorts, var_types)
