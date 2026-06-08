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


def test_eq_elim_does_not_substitute_literal_into_membership_slot():
    """``∃v. (in (v) R) ∧ (= v 5)`` is NOT eliminated.

    Substituting the literal ``5`` into ``v``'s position in
    ``(in (v) R)`` has no DRC representation (positional slots take
    variable names, not literals). If the eliminator dropped the
    binder and the equality, the leftover ``(in (v) R)`` would mention
    ``v`` as a *free* constant — the formula would say "is there any
    row with the same ``v`` as the one bound outside?" instead of
    "is there a row with v=5". The dev_78 BIRD case is the canonical
    instance: ``∃City. schools(..., City, ...) ∧ City = "Adelanto"``
    must NOT collapse to ``schools(..., City, ...)`` with a free
    String constant.

    The eliminator detects this case and skips the rewrite, leaving
    the existential and the equality intact.
    """
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

    # Existential survives; equality survives; semantics preserved.
    assert isinstance(out, QuantifierNode)
    assert out.kind == "exists"
    assert out.variables == ["v"]


def test_eq_elim_substitutes_literal_into_non_slot_reference():
    """When ``v`` only appears in a comparison (no membership slot),
    literal substitution IS safe and the eliminator runs."""
    # ``∃ v. (and (> v 0) (= v 5))`` — ``v`` is in a comparison, not
    # a slot — eliminate it.
    inner = LogicalConnectiveNode(
        operator="and",
        left=ComparisonNode(
            operator=">",
            left=VariableRefNode(name="v"),
            right=LiteralNode(value=0, data_type="number"),
        ),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="v"),
            right=LiteralNode(value=5, data_type="number"),
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["v"], body=inner)

    out = eliminate_trivial_equalities(expr)

    # Existential is gone; ``v`` got substituted with 5 in the comparison.
    assert isinstance(out, ComparisonNode)
    assert out.operator == ">"
    assert isinstance(out.right, LiteralNode)
    assert out.right.value == 0
    assert isinstance(out.left, LiteralNode)
    assert out.left.value == 5


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

    Pass 1 sees ``(= k 7)`` but cannot eliminate it: ``k`` appears as
    a membership slot, and there's no DRC representation for "literal
    at slot k". The existential and equality survive untouched.

    Pass 2 then notices that ``name`` and ``addr`` are bound but never
    referenced anywhere — those slots are pruned. ``k`` IS referenced
    by the equality ``(= k 7)`` so its slot is kept. The result is::

        ∃ k. R(k) ∧ (= k 7)
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
    # Only the ``k`` slot survives — it's referenced by the equality.
    assert membs[0].variables == ["k"]


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


# ---------------------------------------------------------------------------
# Regression tests for the dev_78 false-negative.
#
# Background:
# The runner compares planner-generated DRC vs gold DRC translated from
# BIRD's reference SQL. The generated side starts as
#   ``∃ City, ... . schools(..., City, ...) ∧ City = "Adelanto"``
# while the gold side keeps the equality on a fresh constant
#   ``schools(..., v_schools_city_10, ...) ∧ v_schools_city_10 = "Adelanto"``.
# The old equality eliminator rewrote the generated side to drop the
# binder and the equality, leaving ``schools(..., City, ...)`` with
# ``City`` now a *free constant* whose value the equivalence check
# couldn't tie back to "Adelanto". cvc5 reported ``not_equivalent``
# even though the two formulas describe identical relations.
#
# The fix: skip elimination when the bound variable appears as a
# membership slot AND the would-be replacement is a literal.
# Variable-to-variable substitution stays available because slots
# accept variables.
# ---------------------------------------------------------------------------


def test_dev_78_shape_preserves_equality_on_membership_slot():
    """The exact dev_78 LHS shape: ``∃City. (in (..City..) schools) ∧
    City = "Adelanto"`` must be preserved."""
    inner = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(
            variables=["CDSCode", "City", "GSserved"], relation="schools",
        ),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="City"),
            right=LiteralNode(value="Adelanto", data_type="string"),
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["City"], body=inner)

    out = eliminate_trivial_equalities(expr)

    # The existential survives. We don't pin the exact tree shape
    # below the existential — only that ``City`` is still a bound name
    # (not a free constant) and the equality is still around.
    assert isinstance(out, QuantifierNode)
    assert out.kind == "exists"
    assert "City" in out.variables


def test_dev_78_pair_preprocessing_keeps_both_sides_aligned():
    """Through the FULL pipeline, the generated and gold sides reduce
    to formulas with matching slot signatures.

    This is the regression test for the run-08 dev_78 false negative:
    feeding both sides through ``preprocess_for_smt_pair`` must produce
    membership terms that *both* still mention the City slot, otherwise
    cvc5 sees ill-aligned predicates and reports not-equivalent on
    formulas that are logically the same."""
    # Generated side: ``∃ City, GSserved. schools(CDSCode, City, GSserved)
    #                  ∧ City = "Adelanto"``
    gen = QuantifierNode(
        kind="exists",
        variables=["City", "GSserved"],
        body=LogicalConnectiveNode(
            operator="and",
            left=MembershipNode(
                variables=["CDSCode", "City", "GSserved"],
                relation="schools",
            ),
            right=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="City"),
                right=LiteralNode(value="Adelanto", data_type="string"),
            ),
        ),
    )
    # Gold side: same shape but with the converter's fresh-constant
    # naming convention. The constant ``v_city`` is bound by an
    # existential just like ``City`` on the generated side.
    gold = QuantifierNode(
        kind="exists",
        variables=["v_city", "v_gss"],
        body=LogicalConnectiveNode(
            operator="and",
            left=MembershipNode(
                variables=["v_cds", "v_city", "v_gss"], relation="schools",
            ),
            right=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="v_city"),
                right=LiteralNode(value="Adelanto", data_type="string"),
            ),
        ),
    )

    pre_gen, pre_gold = preprocess_for_smt_pair(
        [gen, gold],
        keep_names={"CDSCode", "v_cds"},
    )

    gen_membs = _gather_memberships(pre_gen)
    gold_membs = _gather_memberships(pre_gold)
    assert len(gen_membs) == 1
    assert len(gold_membs) == 1
    # Both sides keep the same number of slots — the City slot must
    # have survived on the generated side, since it's still constrained
    # by the equality. (The original bug dropped this slot.)
    assert len(gen_membs[0].variables) == len(gold_membs[0].variables)


def test_eq_elim_with_variable_substitution_into_membership_slot_is_allowed():
    """Variable-to-variable substitution is still permitted into
    membership slots — the generated AST is well-formed."""
    # ``∃ a, b. (in (a) R) ∧ (= a b)`` — eliminate ``a`` (or ``b``),
    # collapsing to ``∃ x. (in (x) R)``.
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["a"], relation="R"),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="a"),
            right=VariableRefNode(name="b"),
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["a", "b"], body=body)

    out = eliminate_trivial_equalities(expr)

    # One existential remains, binding either ``a`` or ``b``.
    assert isinstance(out, QuantifierNode)
    assert out.kind == "exists"
    assert len(out.variables) == 1
    membs = _gather_memberships(out)
    assert len(membs) == 1
    assert membs[0].relation == "R"
    # The membership slot now references the surviving binder.
    assert membs[0].variables == out.variables
