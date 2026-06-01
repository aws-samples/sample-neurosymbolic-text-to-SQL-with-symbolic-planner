"""Recognize ``≥N`` patterns and synthesise the canonical "exactly N" form.

This module is used by :func:`apply_difference` to special-case the
common idiom

    (≥N tuples in R sharing key k)  −  (≥N+1 tuples in R sharing key k)
    ────────────────────────────────────────────────────────────────────
                        =  exactly N tuples

The rewrite serves two purposes:

1. It produces a *canonical* "exactly N" DRC form. The pattern emitted
   here is structurally identical to what a hand-written DRC would use
   for "exactly N" — N existentially-bound witnesses with full pairwise
   distinctness, plus a single ``¬∃`` clause asserting "no extra
   witness distinct from each of the N".

2. The canonical form aligns Skolem terms across both sides of an
   equivalence check. Without the rewrite, the difference output is
   a flat ``(C_LHS) ∧ ¬(C_RHS)`` where C_LHS and C_RHS each Skolemise
   their own witnesses; cvc5's E-matching can't unify them and returns
   ``unknown``. With the rewrite, witnesses are shared between the
   positive and negated halves so the existential structure aligns
   with the target's natural shape.

The recogniser is conservative: it returns ``None`` whenever an input
doesn't unambiguously match a ``≥k`` pattern. In particular, missing
pairwise-distinctness or a non-uniform slot pattern across witnesses
causes recognition to fail, falling back to the generic difference
encoding ``C_LHS ∧ ¬C_RHS`` in the caller.

Soundness: the rewrite emits the canonical exactly-N formula

    (C_LHS, unchanged) ∧ ¬∃ w_extra, anon_slots. R(...) ∧ (w_extra ≠ w_i)_i

which is the standard FOL encoding of "≥N tuples ∧ ¬(≥N+1 tuples)".
The encoding is a tautology of the input pair (C_LHS = ≥N, C_RHS =
≥N+1) given that the recogniser guarantees both sides share the same
(relation, witness slot, key slots) shape.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from text_to_sql_planner.types.drc import (
    ColumnVariable,
    ComparisonNode,
    DRCCondition,
    DRCExpression,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)


# ---------------------------------------------------------------------------
# Pattern recognition
# ---------------------------------------------------------------------------


@dataclass
class GeNPattern:
    """A recognised ``≥N tuples in R sharing key`` pattern.

    Attributes:
        relation: The relation name R that the witnesses range over.
        witness_slot: Slot index (0-based) of the per-witness column in
            R's argument list.
        key_slots: Mapping ``slot -> result-variable name``. These
            slots are pinned to the result variables (the "key"
            columns the witnesses must agree on, e.g. ``emp_id``).
        slot_arity: Total number of slots in R (used to reconstruct
            membership nodes with the same arity).
        witnesses: The N witness variable names, in deterministic
            order (sorted lexicographically).
        template_membership: One representative ``MembershipNode``
            from the input — its slot pattern is reused (with fresh
            anonymous names) when synthesising the "extra witness"
            membership for the canonical form.
    """

    relation: str
    witness_slot: int
    key_slots: dict[int, str]
    slot_arity: int
    witnesses: list[str]
    template_membership: MembershipNode


def recognize_ge_n_pattern(expr: DRCExpression) -> GeNPattern | None:
    """Return a :class:`GeNPattern` if ``expr`` encodes ``≥N tuples in
    R sharing key columns``, else ``None``.

    Recognition rules (all must hold):

    * The result variables of ``expr`` are non-empty plain
      :class:`ColumnVariable` references (no aggregates).
    * Walking through nested existentials and conjunctions, there are
      *N ≥ 2* membership nodes against the same relation R such that:

      - The same set of slots in R are pinned to result variables
        (the "key slots"), with the same result-variable name at
        each slot.
      - One slot is occupied by a quantifier-bound variable (the
        witness) that differs across the N memberships.
      - The N witnesses are pairwise distinct under top-level
        ``!=`` constraints reachable through ``∃`` and ``∧``.
    """
    result_vars = _result_var_names(expr)
    if not result_vars:
        return None

    conjuncts: list[tuple[frozenset[str], DRCCondition]] = []
    distinctness: set[frozenset[str]] = set()
    _collect_top_level(expr.condition, frozenset(), conjuncts, distinctness)

    # Bucket every membership-of-some-relation that has at least one
    # result-variable slot and one bound-variable slot.
    candidate_groups: dict[
        tuple[str, int, frozenset[tuple[int, str]], int],
        list[tuple[str, MembershipNode]],
    ] = defaultdict(list)

    for bound, node in conjuncts:
        if not isinstance(node, MembershipNode):
            continue

        # Slots split into three roles:
        # - key slots: pinned to a result variable
        # - witness slot: occupied by a bound variable that's not also
        #   a result variable
        # - other slots: ignored at recognition time (anonymous slots
        #   that get rebound per-witness)
        key_slots: dict[int, str] = {}
        witness_candidates: list[tuple[int, str]] = []
        for slot, var in enumerate(node.variables):
            if var in result_vars:
                key_slots[slot] = var
            elif var in bound:
                witness_candidates.append((slot, var))

        if not key_slots or not witness_candidates:
            continue

        # If multiple bound variables sit in this membership, we don't
        # know which is "the witness" — skip. (In practice ≥N
        # patterns from the planner have exactly one bound non-key
        # slot per membership.)
        if len(witness_candidates) != 1:
            continue
        w_slot, w_var = witness_candidates[0]

        key_signature = frozenset(key_slots.items())
        group_key = (node.relation, w_slot, key_signature, len(node.variables))
        candidate_groups[group_key].append((w_var, node))

    if not candidate_groups:
        return None

    # Pick the group with the most distinct witnesses. Ties go to the
    # first relation seen.
    best_key = max(
        candidate_groups.keys(),
        key=lambda k: len({w for w, _ in candidate_groups[k]}),
    )
    members = candidate_groups[best_key]
    relation, w_slot, key_signature, arity = best_key

    # De-duplicate witnesses (the same witness can appear in more than
    # one membership conjunct, e.g. when the planner emits a ≥N pattern
    # with an extra Performance_Reviews lookup for join-on-emp_id).
    seen_witnesses: list[str] = []
    seen_set: set[str] = set()
    for w_var, _node in members:
        if w_var not in seen_set:
            seen_set.add(w_var)
            seen_witnesses.append(w_var)

    if len(seen_witnesses) < 2:
        return None

    # Verify pairwise distinctness across all (i, j).
    for i, w1 in enumerate(seen_witnesses):
        for w2 in seen_witnesses[i + 1 :]:
            if frozenset({w1, w2}) not in distinctness:
                return None

    template = members[0][1]
    return GeNPattern(
        relation=relation,
        witness_slot=w_slot,
        key_slots=dict(key_signature),
        slot_arity=arity,
        witnesses=sorted(seen_witnesses),
        template_membership=template,
    )


# ---------------------------------------------------------------------------
# Canonical "exactly N" emission
# ---------------------------------------------------------------------------


def emit_canonical_exactly_n(
    expr: DRCExpression, pattern: GeNPattern,
) -> DRCExpression:
    """Build ``expr ∧ ¬∃ w_extra, anon_slots. R(...) ∧ ⋀_i (w_extra ≠ w_i)``.

    The synthesised membership reuses the slot pattern of
    ``pattern.template_membership`` — key slots keep their result-
    variable name, the witness slot is filled with a fresh ``w_extra``
    name, and other (anonymous) slots get fresh per-position names so
    they don't capture any name in scope.
    """
    used_names = _all_names_in_expression(expr)

    w_extra = _fresh_name(used_names, hint="w_extra")
    used_names.add(w_extra)

    # Slot pattern: witness slot -> w_extra; key slots -> result vars;
    # other slots -> fresh anon names.
    new_slots: list[str] = []
    fresh_anons: list[str] = []
    for slot in range(pattern.slot_arity):
        if slot == pattern.witness_slot:
            new_slots.append(w_extra)
        elif slot in pattern.key_slots:
            new_slots.append(pattern.key_slots[slot])
        else:
            anon = _fresh_name(used_names, hint=f"anon{slot}")
            used_names.add(anon)
            fresh_anons.append(anon)
            new_slots.append(anon)

    membership = MembershipNode(
        variables=new_slots, relation=pattern.relation,
    )

    # Build (w_extra != w_1) ∧ (w_extra != w_2) ∧ ... left-associatively.
    distinctness: DRCCondition | None = None
    for wi in pattern.witnesses:
        ineq = ComparisonNode(
            operator="!=",
            left=VariableRefNode(name=w_extra),
            right=VariableRefNode(name=wi),
        )
        distinctness = (
            ineq
            if distinctness is None
            else LogicalConnectiveNode(
                operator="and", left=distinctness, right=ineq,
            )
        )

    if distinctness is None:
        body: DRCCondition = membership
    else:
        body = LogicalConnectiveNode(
            operator="and", left=membership, right=distinctness,
        )

    inner_existential = QuantifierNode(
        kind="exists",
        variables=[w_extra] + fresh_anons,
        body=body,
    )

    no_extra_witness = NotNode(operand=inner_existential)
    new_condition = LogicalConnectiveNode(
        operator="and", left=expr.condition, right=no_extra_witness,
    )

    return DRCExpression(
        result_variables=list(expr.result_variables),
        condition=new_condition,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_var_names(expr: DRCExpression) -> set[str]:
    """Return the set of column-variable names among ``expr``'s result
    variables. Aggregate result variables don't participate in pattern
    recognition (they aren't keys for ≥N over a relation), so they're
    silently excluded.
    """
    out: set[str] = set()
    for rv in expr.result_variables:
        if isinstance(rv, ColumnVariable):
            out.add(rv.name)
    return out


def _collect_top_level(
    node: DRCCondition,
    bound: frozenset[str],
    conjuncts: list[tuple[frozenset[str], DRCCondition]],
    distinctness: set[frozenset[str]],
) -> None:
    """Walk through chains of ``∃`` and ``∧`` collecting:

    * leaf conjuncts (paired with the set of variables bound at that
      depth), and
    * pairwise ``!=`` constraints between bound variables (as
      frozensets, regardless of operand order).

    Stops at ``∨`` / ``¬`` / ``→`` / ``∀``: those are returned as
    opaque conjuncts. We don't attempt to recognise ``≥N`` patterns
    that span those constructs — the planner's ≥N output is always a
    pure ``∃ ... ∧ ...`` chain.
    """
    if node is None:
        return

    if isinstance(node, QuantifierNode) and node.kind == "exists":
        new_bound = bound | frozenset(node.variables)
        _collect_top_level(node.body, new_bound, conjuncts, distinctness)
        return

    if isinstance(node, LogicalConnectiveNode) and node.operator == "and":
        _collect_top_level(node.left, bound, conjuncts, distinctness)
        _collect_top_level(node.right, bound, conjuncts, distinctness)
        return

    if isinstance(node, ComparisonNode) and node.operator == "!=":
        if (
            isinstance(node.left, VariableRefNode)
            and isinstance(node.right, VariableRefNode)
            and node.left.name in bound
            and node.right.name in bound
            and node.left.name != node.right.name
        ):
            distinctness.add(frozenset({node.left.name, node.right.name}))
        # Fall through: still record this as a conjunct so callers
        # that don't care about distinctness can see it.
        conjuncts.append((bound, node))
        return

    conjuncts.append((bound, node))


def _all_names_in_expression(expr: DRCExpression) -> set[str]:
    """Return every variable name that appears in ``expr`` — free
    references, bound variables, membership slots, result variables.
    Used by ``_fresh_name`` to avoid capture.
    """
    names: set[str] = set()
    for rv in expr.result_variables:
        if isinstance(rv, ColumnVariable):
            names.add(rv.name)
        elif hasattr(rv, "column"):
            names.add(rv.column)
    _walk_collect_names(expr.condition, names)
    return names


def _walk_collect_names(node: DRCCondition, names: set[str]) -> None:
    if node is None:
        return
    if isinstance(node, VariableRefNode):
        names.add(node.name)
        return
    if isinstance(node, MembershipNode):
        names.update(node.variables)
        return
    if isinstance(node, QuantifierNode):
        names.update(node.variables)
        _walk_collect_names(node.body, names)
        return
    if isinstance(node, LogicalConnectiveNode):
        _walk_collect_names(node.left, names)
        _walk_collect_names(node.right, names)
        return
    if isinstance(node, NotNode):
        _walk_collect_names(node.operand, names)
        return
    if isinstance(node, ComparisonNode):
        _walk_collect_names(node.left, names)
        _walk_collect_names(node.right, names)
        return
    # Other node types (Arithmetic, FunctionCall, Literal) recurse
    # similarly; the recogniser only looks at membership/comparison
    # patterns so we don't bother with the others here. Names that
    # might appear inside arithmetic etc. are still collected via
    # the generic ``children`` traversal below.
    children = getattr(node, "arguments", None)
    if children:
        for c in children:
            _walk_collect_names(c, names)
        return
    for attr in ("left", "right", "operand", "body"):
        c = getattr(node, attr, None)
        if c is not None:
            _walk_collect_names(c, names)


def _fresh_name(used: set[str], *, hint: str) -> str:
    """Allocate a name not in ``used``. Tries ``_<hint>`` first,
    then falls back to numeric suffixes ``_<hint>_0, _<hint>_1, …``.
    """
    candidate = f"_{hint}"
    if candidate not in used:
        return candidate
    i = 0
    while True:
        candidate = f"_{hint}_{i}"
        if candidate not in used:
            return candidate
        i += 1
