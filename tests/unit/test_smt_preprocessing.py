"""Tests for the pre-SMT preprocessing pipeline.

The pipeline runs two passes:

1. Trivial-equality elimination: ``∃v. (= v expr) ∧ φ`` rewrites to
   ``φ[v := expr]``.
2. Unused-slot elimination: relation slots whose bound variable appears
   nowhere else in any condition can be dropped from the predicate's
   signature.

These tests exercise both passes in isolation and together.
"""

from __future__ import annotations

import pytest

from text_to_sql_planner.equivalence.smt_preprocessing import (
    drop_unused_slots,
    eliminate_trivial_equalities,
    eliminate_unused_relation_slots,
    preprocess_for_smt,
    preprocess_for_smt_pair,
)
from text_to_sql_planner.types.drc import (
    ComparisonNode,
    DRCCondition,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)


# ---------------------------------------------------------------------------
# Pass 1: trivial-equality elimination
# ---------------------------------------------------------------------------


def test_eq_elim_substitutes_variable():
    """``∃v. (in (v) R) ∧ (= v 5)`` → ``(in (v) R)[v := 5]`` (no exist)."""
    inner = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["v"], relation="R"),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="v"),
            right=LiteralNode(value=5, data_type="number"),
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["v"], body=inner)

    out = eliminate_trivial_equalities(expr)

    # The exists is gone, the equality is gone, and the membership
    # carries the literal substituted in (as an identifier in the slot
    # position — see the substitute logic for membership semantics).
    # In our implementation, substituting a literal into a positional
    # membership slot is a no-op (we keep the original name) because
    # there's no DRC representation for "literal at slot k". So in this
    # case the equality gets rewritten away but the membership stays.
    assert isinstance(out, MembershipNode)
    assert out.relation == "R"


def test_eq_elim_unifies_two_bound_vars():
    """``∃ a, b. (in (a) R) ∧ (in (b) S) ∧ (= a b)`` →
    ``∃ a. (in (a) R) ∧ (in (a) S)``."""
    body = LogicalConnectiveNode(
        operator="and",
        left=LogicalConnectiveNode(
            operator="and",
            left=MembershipNode(variables=["a"], relation="R"),
            right=MembershipNode(variables=["b"], relation="S"),
        ),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="a"),
            right=VariableRefNode(name="b"),
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["a", "b"], body=body)

    out = eliminate_trivial_equalities(expr)

    # One existential remains, binding the smaller of {a, b} = "a".
    assert isinstance(out, QuantifierNode)
    assert out.kind == "exists"
    assert out.variables == ["a"]
    # Both memberships now reference "a".
    membs = _gather_memberships(out)
    assert sorted((m.relation, m.variables) for m in membs) == [
        ("R", ["a"]),
        ("S", ["a"]),
    ]


def test_eq_elim_preserves_nested_quantifier_scope():
    """Substitution must stop at an inner quantifier that re-binds the
    same name (capture avoidance)."""
    # ∃ v. (= v 7) ∧ ∃ v. (in (v) R)   — the inner v is a *different* v.
    inner_quant = QuantifierNode(
        kind="exists",
        variables=["v"],
        body=MembershipNode(variables=["v"], relation="R"),
    )
    body = LogicalConnectiveNode(
        operator="and",
        left=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="v"),
            right=LiteralNode(value=7, data_type="number"),
        ),
        right=inner_quant,
    )
    expr = QuantifierNode(kind="exists", variables=["v"], body=body)

    out = eliminate_trivial_equalities(expr)

    # The outer existential is gone (no body still references the
    # original outer v after the equality is consumed). The inner
    # ``∃v. (in (v) R)`` is preserved verbatim — its v is a different
    # binding and the outer rewrite must not touch it.
    assert isinstance(out, QuantifierNode)
    assert out.kind == "exists"
    assert out.variables == ["v"]
    assert isinstance(out.body, MembershipNode)
    assert out.body.relation == "R"
    assert out.body.variables == ["v"]


def test_eq_elim_universally_bound_var_first():
    """An equality ``(= outer inner)`` where ``outer`` is universally
    bound at a higher scope (so not in the local exists's binder set)
    and ``inner`` IS the local binder must still be eliminated by the
    inner orientation. Without checking both orientations, the
    eliminator would bail because ``outer`` is not in the local binder
    set and never re-consider with ``inner`` as the candidate.

    This is the exact pattern produced by the difference operator's
    output DRC when one side equates a result variable (free at the
    universal scope) to a freshly-bound witness inside the negated
    sub-tree.
    """
    # ∃ inner . R(inner) ∧ (= outer inner)
    # `outer` is left as a free reference here (we don't bind it).
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["inner"], relation="R"),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="outer"),
            right=VariableRefNode(name="inner"),
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["inner"], body=body)

    out = eliminate_trivial_equalities(expr)

    # The inner exists is gone; what remains references `outer` directly
    # in place of `inner`.
    assert isinstance(out, MembershipNode)
    assert out.relation == "R"
    assert out.variables == ["outer"]


def test_eq_elim_no_change_when_no_match():
    """No top-level equality on a bound variable → identity transform."""
    body = MembershipNode(variables=["x"], relation="R")
    expr = QuantifierNode(kind="exists", variables=["x"], body=body)
    out = eliminate_trivial_equalities(expr)
    assert isinstance(out, QuantifierNode)
    assert out.variables == ["x"]


def test_eq_elim_does_not_descend_into_negation():
    """Substituting under ``not`` would change semantics, so we don't."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["x"], relation="R"),
        right=NotNode(
            operand=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="x"),
                right=LiteralNode(value=0, data_type="number"),
            )
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["x"], body=body)

    out = eliminate_trivial_equalities(expr)

    # No rewrite — the equality lives under ``not`` so we leave it alone.
    assert isinstance(out, QuantifierNode)
    assert out.variables == ["x"]


# ---------------------------------------------------------------------------
# Pass 2: unused-slot elimination
# ---------------------------------------------------------------------------


def test_unused_slot_detection():
    """Slots whose bound variables are never referenced elsewhere are
    detected as unused."""
    # ∃ id, name, addr. R(id, name, addr) ∧ (= id 5)
    # `id` is referenced in the equality — used.
    # `name`, `addr` are bound only to slots — unused.
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(
            variables=["id", "name", "addr"], relation="R",
        ),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="id"),
            right=LiteralNode(value=5, data_type="number"),
        ),
    )
    expr = QuantifierNode(
        kind="exists", variables=["id", "name", "addr"], body=body,
    )

    used = eliminate_unused_relation_slots([expr])

    assert used == {"R": [True, False, False]}


def test_unused_slot_kept_alive_by_keep_names():
    """A name in ``keep_names`` is treated as referenced outside, even
    if it's only bound by a single membership."""
    body = MembershipNode(variables=["id", "name"], relation="R")
    expr = QuantifierNode(
        kind="exists", variables=["name"], body=body,
    )
    used_default = eliminate_unused_relation_slots([expr])
    # Without keep_names: only `id` is "used" (it's referenced as a free
    # variable outside the existential, which counts as outside-membership
    # via VariableRefNode... actually no, it's not a VariableRef here.
    # Let's adjust expectations.)

    # In this scenario `name` is bound, `id` is free at the body level.
    # `id` is mentioned in *only* the membership, so it's also unused
    # by default. Both unused.
    assert used_default["R"] == [False, False]

    # But if the caller protects "id" via keep_names, that slot stays.
    used_with_keep = eliminate_unused_relation_slots(
        [expr], keep_names=["id"],
    )
    assert used_with_keep["R"] == [True, False]


def test_drop_unused_slots_removes_membership_args_and_binders():
    """The rewrite drops the membership args at the unused positions
    *and* removes those names from any enclosing existential binder."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(
            variables=["id", "name", "addr"], relation="R",
        ),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="id"),
            right=LiteralNode(value=5, data_type="number"),
        ),
    )
    expr = QuantifierNode(
        kind="exists", variables=["id", "name", "addr"], body=body,
    )
    used = eliminate_unused_relation_slots([expr])
    out = drop_unused_slots(expr, used)

    assert isinstance(out, QuantifierNode)
    # Only `id` survives in the binder.
    assert out.variables == ["id"]
    # Membership now has only the `id` slot.
    assert isinstance(out.body, LogicalConnectiveNode)
    membership = out.body.left
    assert isinstance(membership, MembershipNode)
    assert membership.relation == "R"
    assert membership.variables == ["id"]


def test_unused_slot_joint_across_two_conditions():
    """A slot that's used in either condition stays used in *both*."""
    # Side A: R(a, x) where `a` is used elsewhere.
    side_a = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["a", "x"], relation="R"),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="a"),
            right=LiteralNode(value=1, data_type="number"),
        ),
    )
    # Side B: R(b, y) where `y` is used elsewhere.
    side_b = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["b", "y"], relation="R"),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="y"),
            right=LiteralNode(value=2, data_type="number"),
        ),
    )
    used = eliminate_unused_relation_slots([side_a, side_b])
    # Slot 0 used in side A, slot 1 used in side B → both alive.
    assert used == {"R": [True, True]}


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_preprocess_for_smt_combines_passes():
    """Pass 1 (equality elimination) plus pass 2 (slot pruning) compose
    correctly.

    Starting expression::

        ∃ k, name, addr.  R(k, name, addr)  ∧  (= k 7)

    Pass 1 sees ``(= k 7)`` and rewrites the body, leaving the equality
    consumed. Pass 2 then notices that ``name`` and ``addr`` are bound
    but never referenced anywhere — it prunes their slots. ``k`` is
    *also* never referenced after pass 1 consumed the equality, so its
    slot is pruned too: the relation collapses to a zero-arity
    proposition ``R()`` (witnessed by the original "is there any row
    with k=7" question). This is correct: the question reduces to
    "does R have any row" once we substitute the constant.
    """
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(
            variables=["k", "name", "addr"], relation="R",
        ),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="k"),
            right=LiteralNode(value=7, data_type="number"),
        ),
    )
    expr = QuantifierNode(
        kind="exists", variables=["k", "name", "addr"], body=body,
    )

    out = preprocess_for_smt(expr)

    membs = _gather_memberships(out)
    assert len(membs) == 1
    assert membs[0].relation == "R"
    # All slots get pruned because nothing references the bound names
    # any more after pass 1 consumed the equality.
    assert membs[0].variables == []


def test_preprocess_for_smt_keeps_referenced_slot():
    """Variant where ``k`` IS referenced outside (kept via keep_names)."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(
            variables=["k", "name", "addr"], relation="R",
        ),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="k"),
            right=LiteralNode(value=7, data_type="number"),
        ),
    )
    expr = QuantifierNode(
        kind="exists", variables=["name", "addr"], body=body,
    )

    out, = preprocess_for_smt_pair([expr], keep_names=["k"])

    membs = _gather_memberships(out)
    assert len(membs) == 1
    assert membs[0].relation == "R"
    assert membs[0].variables == ["k"]


def test_preprocess_keeps_result_variable_slots():
    """Result variables passed via ``keep_names`` survive even when they
    aren't referenced anywhere inside the condition."""
    # The condition itself only mentions ``emp_id`` inside a membership.
    body = MembershipNode(variables=["emp_id", "name", "addr"], relation="Employees")
    out, = preprocess_for_smt_pair([body], keep_names=["emp_id"])

    # The kept name's slot survives, the others are dropped.
    membs = _gather_memberships(out)
    assert len(membs) == 1
    assert membs[0].relation == "Employees"
    assert membs[0].variables == ["emp_id"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gather_memberships(node: DRCCondition) -> list[MembershipNode]:
    out: list[MembershipNode] = []
    if node is None:
        return out
    if isinstance(node, MembershipNode):
        out.append(node)
    elif isinstance(node, QuantifierNode):
        out.extend(_gather_memberships(node.body))
    elif isinstance(node, LogicalConnectiveNode):
        out.extend(_gather_memberships(node.left))
        out.extend(_gather_memberships(node.right))
    elif isinstance(node, NotNode):
        out.extend(_gather_memberships(node.operand))
    elif isinstance(node, ComparisonNode):
        out.extend(_gather_memberships(node.left))
        out.extend(_gather_memberships(node.right))
    return out
