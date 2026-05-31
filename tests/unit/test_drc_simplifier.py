"""Tests for the DRC expression simplifier.

The simplifier runs five rewrite passes to a fixed point:

1. merge nested quantifiers,
2. eliminate trivial equalities,
3. boolean simplification,
4. push existentials inward (minimise scope),
5. drop unused binders.

These tests exercise each pass in isolation plus a few interaction
tests where multiple passes need to compose to reach the final form.
"""

from __future__ import annotations

import pytest

from text_to_sql_planner.drc_simplifier import simplify_drc
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    ComparisonNode,
    DRCCondition,
    DRCExpression,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expr(condition: DRCCondition, result_vars: list[str] | None = None) -> DRCExpression:
    rvs = [ColumnVariable(name=n) for n in (result_vars or ["x"])]
    return DRCExpression(result_variables=rvs, condition=condition)


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


def _all_quantifier_binders(node: DRCCondition) -> list[list[str]]:
    out: list[list[str]] = []
    if node is None:
        return out
    if isinstance(node, QuantifierNode):
        out.append(list(node.variables))
        out.extend(_all_quantifier_binders(node.body))
    elif isinstance(node, LogicalConnectiveNode):
        out.extend(_all_quantifier_binders(node.left))
        out.extend(_all_quantifier_binders(node.right))
    elif isinstance(node, NotNode):
        out.extend(_all_quantifier_binders(node.operand))
    elif isinstance(node, ComparisonNode):
        out.extend(_all_quantifier_binders(node.left))
        out.extend(_all_quantifier_binders(node.right))
    return out


# ---------------------------------------------------------------------------
# Pass 1: merge nested quantifiers
# ---------------------------------------------------------------------------


def test_merge_nested_existentials():
    """``∃ a. ∃ b. ∃ c. φ → ∃ a, b, c. φ``."""
    body = MembershipNode(variables=["a", "b", "c"], relation="R")
    inner = QuantifierNode(kind="exists", variables=["c"], body=body)
    middle = QuantifierNode(kind="exists", variables=["b"], body=inner)
    outer = QuantifierNode(kind="exists", variables=["a"], body=middle)

    out = simplify_drc(_expr(outer)).condition

    assert isinstance(out, QuantifierNode)
    assert out.kind == "exists"
    # Three bound variables fused into one quantifier (alpha-renamed
    # to a deterministic ``_v0, _v1, _v2`` sequence).
    assert len(out.variables) == 3
    assert isinstance(out.body, MembershipNode)


def test_does_not_merge_different_kinds():
    """``∃ a. ∀ b. φ`` is NOT mergeable."""
    body = MembershipNode(variables=["a", "b"], relation="R")
    inner = QuantifierNode(kind="forall", variables=["b"], body=body)
    outer = QuantifierNode(kind="exists", variables=["a"], body=inner)

    out = simplify_drc(_expr(outer)).condition

    binders = _all_quantifier_binders(out)
    # Two separate binders, kinds preserved (sizes match the original).
    assert [len(b) for b in binders] == [1, 1]
    # Outer kind is exists, inner is forall.
    assert isinstance(out, QuantifierNode)
    assert out.kind == "exists"
    inner_node = out.body
    assert isinstance(inner_node, QuantifierNode)
    assert inner_node.kind == "forall"


# ---------------------------------------------------------------------------
# Pass 2: trivial-equality elimination
# ---------------------------------------------------------------------------


def test_eq_elim_unifies_two_bound_variables():
    """``∃ a, b. R(a) ∧ S(b) ∧ (= a b) → ∃ a. R(a) ∧ S(a)`` (smaller name wins).

    After alpha-canonicalisation both memberships will reference the
    same single bound name, but the *exact* name is the deterministic
    ``_v0`` (or whatever the canonicaliser assigned). What we check is
    that both memberships agree on it.
    """
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

    out = simplify_drc(_expr(expr)).condition

    membs = _gather_memberships(out)
    assert len(membs) == 2
    # Both single-slot memberships now agree on the same bound name.
    relations = sorted(m.relation for m in membs)
    assert relations == ["R", "S"]
    assert membs[0].variables == membs[1].variables
    assert len(membs[0].variables) == 1


def test_eq_elim_skips_literal_into_membership_slot():
    """When the substituent is a literal AND the bound variable appears
    in a membership slot, the rewrite must NOT fire — there's no DRC
    representation for "literal at slot k", and dropping the binder
    would leave a dangling slot reference."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["k"], relation="R"),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="k"),
            right=LiteralNode(value=7, data_type="number"),
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["k"], body=body)

    out = simplify_drc(_expr(expr)).condition

    # The existential and the equality both survive — the bound name
    # gets alpha-renamed but the structural shape is preserved.
    assert isinstance(out, QuantifierNode)
    assert len(out.variables) == 1
    membs = _gather_memberships(out)
    assert membs[0].relation == "R"
    # The membership's slot still binds the (renamed) name.
    assert membs[0].variables == out.variables


def test_eq_elim_does_not_descend_into_negation():
    """Substituting under a ``not`` would change semantics."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["x"], relation="R"),
        right=NotNode(
            operand=ComparisonNode(
                operator="=",
                left=VariableRefNode(name="x"),
                right=LiteralNode(value=0, data_type="number"),
            ),
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["x"], body=body)

    out = simplify_drc(_expr(expr)).condition

    # No rewrite — the equality is under not. The existential survives
    # (bound variable alpha-renamed deterministically).
    # NNF rewrites ¬(= x 0) to (!= x 0), which doesn't change scope.
    assert isinstance(out, QuantifierNode)
    assert len(out.variables) == 1


# ---------------------------------------------------------------------------
# Pass 3: boolean simplification
# ---------------------------------------------------------------------------


def test_boolean_double_negation():
    """``¬¬R(x) → R(x)``."""
    body = NotNode(
        operand=NotNode(operand=MembershipNode(variables=["x"], relation="R"))
    )
    expr = QuantifierNode(kind="exists", variables=["x"], body=body)

    out = simplify_drc(_expr(expr)).condition

    # After ¬¬ stripping, body is the bare membership; existential
    # over a single occurrence stays.
    assert isinstance(out, QuantifierNode)
    inner = out.body
    assert isinstance(inner, MembershipNode)
    assert inner.relation == "R"


def test_boolean_idempotent_and():
    """``φ ∧ φ → φ``."""
    body = MembershipNode(variables=["x"], relation="R")
    expr = LogicalConnectiveNode(operator="and", left=body, right=body)

    out = simplify_drc(_expr(expr)).condition

    assert isinstance(out, MembershipNode)
    assert out.relation == "R"


def test_boolean_idempotent_or():
    """``φ ∨ φ → φ``."""
    body = MembershipNode(variables=["x"], relation="R")
    expr = LogicalConnectiveNode(operator="or", left=body, right=body)

    out = simplify_drc(_expr(expr)).condition

    assert isinstance(out, MembershipNode)


# ---------------------------------------------------------------------------
# Pass 4: push existentials inward
# ---------------------------------------------------------------------------


def test_push_existential_into_right_conjunct():
    """``∃ x. (R(y) ∧ S(x))`` should push ``∃ x`` into the S branch only."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["y"], relation="R"),
        right=MembershipNode(variables=["x"], relation="S"),
    )
    expr = QuantifierNode(kind="exists", variables=["x"], body=body)

    out = simplify_drc(_expr(expr, ["y"])).condition

    # Top-level should be an AND with R(y) on one side (the kept free
    # ``y`` is a result variable) and ∃ <renamed>. S on the other.
    assert isinstance(out, LogicalConnectiveNode)
    assert out.operator == "and"
    parts = [out.left, out.right]
    has_r = any(
        isinstance(p, MembershipNode)
        and p.relation == "R"
        and p.variables == ["y"]
        for p in parts
    )
    has_quant_s = any(
        isinstance(p, QuantifierNode)
        and p.kind == "exists"
        and len(p.variables) == 1
        and isinstance(p.body, MembershipNode)
        and p.body.relation == "S"
        for p in parts
    )
    assert has_r and has_quant_s


def test_push_splits_disjoint_binders():
    """``∃ x, y. (R(x) ∧ S(y))`` → ``(∃ x. R(x)) ∧ (∃ y. S(y))``."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["x"], relation="R"),
        right=MembershipNode(variables=["y"], relation="S"),
    )
    expr = QuantifierNode(kind="exists", variables=["x", "y"], body=body)

    out = simplify_drc(_expr(expr, ["z"])).condition

    # Final shape: AND of two single-binder existentials over single
    # memberships.
    assert isinstance(out, LogicalConnectiveNode)
    assert out.operator == "and"
    for side in (out.left, out.right):
        assert isinstance(side, QuantifierNode)
        assert side.kind == "exists"
        assert len(side.variables) == 1
        assert isinstance(side.body, MembershipNode)


def test_push_keeps_shared_binders_outside():
    """A binder that's free in *both* conjuncts cannot be pushed."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["x"], relation="R"),
        right=MembershipNode(variables=["x"], relation="S"),
    )
    expr = QuantifierNode(kind="exists", variables=["x"], body=body)

    out = simplify_drc(_expr(expr, ["z"])).condition

    # Outer existential survives unchanged because the bound variable
    # is needed by both conjuncts. Single-binder, alpha-renamed.
    assert isinstance(out, QuantifierNode)
    assert len(out.variables) == 1


# ---------------------------------------------------------------------------
# Pass 5: drop unused binders
# ---------------------------------------------------------------------------


def test_drop_unused_binder_in_existential():
    """``∃ x, y. R(x) → ∃ x. R(x)``."""
    body = MembershipNode(variables=["x"], relation="R")
    expr = QuantifierNode(kind="exists", variables=["x", "y"], body=body)

    out = simplify_drc(_expr(expr)).condition

    assert isinstance(out, QuantifierNode)
    # Only one binder survives (alpha-renamed to a deterministic name).
    assert len(out.variables) == 1


def test_drop_existential_with_no_used_binders():
    """``∃ x. R(y) → R(y)``."""
    body = MembershipNode(variables=["y"], relation="R")
    expr = QuantifierNode(kind="exists", variables=["x"], body=body)

    out = simplify_drc(_expr(expr, ["y"])).condition

    assert isinstance(out, MembershipNode)
    assert out.relation == "R"


# ---------------------------------------------------------------------------
# Interaction tests
# ---------------------------------------------------------------------------


def test_eq_elim_then_push_existentials():
    """A real-world-ish shape: difference operator emits
    ``∃ emp_id. R(emp_id, ...) ∧ ∃ ... S(...) ∧ (= emp_id _rv_0)``.

    After eq-elim + push, ``emp_id`` is unified with ``_rv_0`` and the
    inner existentials end up scoped tightly around the part that
    actually needs them.
    """
    # ∃ emp_id. (in (emp_id) Employees) ∧ ∃ rid. ((in (rid emp_id) PR) ∧ (= emp_id _rv_0))
    inner_pr = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["rid", "emp_id"], relation="PR"),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="emp_id"),
            right=VariableRefNode(name="_rv_0"),
        ),
    )
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["emp_id"], relation="Employees"),
        right=QuantifierNode(kind="exists", variables=["rid"], body=inner_pr),
    )
    expr = QuantifierNode(kind="exists", variables=["emp_id"], body=body)

    out = simplify_drc(_expr(expr, ["_rv_0"])).condition

    # `emp_id` should be substituted away in favor of `_rv_0`. Every
    # membership now references `_rv_0` instead of `emp_id`.
    membs = _gather_memberships(out)
    for m in membs:
        if m.relation == "Employees":
            assert m.variables == ["_rv_0"]
        elif m.relation == "PR":
            assert "_rv_0" in m.variables
            assert "emp_id" not in m.variables


def test_simplifier_is_idempotent():
    """Running the simplifier twice yields the same result as once."""
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

    once = simplify_drc(_expr(expr))
    twice = simplify_drc(once)

    assert _gather_memberships(once.condition) == _gather_memberships(twice.condition)
    assert _all_quantifier_binders(once.condition) == _all_quantifier_binders(twice.condition)


# ---------------------------------------------------------------------------
# Reflexive comparison
# ---------------------------------------------------------------------------


def test_reflexive_equality_collapses_to_true():
    """``(= x x) ∧ R(x) → R(x)`` (the True from reflexivity is pruned
    by the boolean simplifier)."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["x"], relation="R"),
        right=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="x"),
            right=VariableRefNode(name="x"),
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["x"], body=body)

    out = simplify_drc(_expr(expr)).condition

    # Only the membership survives (under an existential binder).
    assert isinstance(out, QuantifierNode)
    assert isinstance(out.body, MembershipNode)
    assert out.body.relation == "R"


def test_reflexive_inequality_collapses_to_false():
    """``(!= x x) ∧ R(x) → False`` (and the AND collapses to False)."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["x"], relation="R"),
        right=ComparisonNode(
            operator="!=",
            left=VariableRefNode(name="x"),
            right=VariableRefNode(name="x"),
        ),
    )
    expr = QuantifierNode(kind="exists", variables=["x"], body=body)

    out = simplify_drc(_expr(expr)).condition

    # The body collapses to False; the existential over False is also
    # False (boolean simplification covers ``and ... False → False``,
    # then the outer existential is dropped because its body is the
    # false-literal).
    assert isinstance(out, LiteralNode)
    assert out.value == 0  # _FALSE marker


# ---------------------------------------------------------------------------
# Negation normal form (De Morgan, quantifier flip)
# ---------------------------------------------------------------------------


def test_nnf_double_negation_eliminated():
    """``¬¬R(x) → R(x)``."""
    body = NotNode(operand=NotNode(operand=MembershipNode(variables=["x"], relation="R")))
    out = simplify_drc(_expr(body, ["x"])).condition
    assert isinstance(out, MembershipNode)
    assert out.relation == "R"


def test_nnf_de_morgan_and():
    """``¬(R(x) ∧ S(x)) → ¬R(x) ∨ ¬S(x)``."""
    body = NotNode(
        operand=LogicalConnectiveNode(
            operator="and",
            left=MembershipNode(variables=["x"], relation="R"),
            right=MembershipNode(variables=["x"], relation="S"),
        ),
    )
    out = simplify_drc(_expr(body, ["x"])).condition

    assert isinstance(out, LogicalConnectiveNode)
    assert out.operator == "or"
    # Each side is a ¬membership.
    assert isinstance(out.left, NotNode)
    assert isinstance(out.right, NotNode)


def test_nnf_de_morgan_or():
    """``¬(R(x) ∨ S(x)) → ¬R(x) ∧ ¬S(x)``."""
    body = NotNode(
        operand=LogicalConnectiveNode(
            operator="or",
            left=MembershipNode(variables=["x"], relation="R"),
            right=MembershipNode(variables=["x"], relation="S"),
        ),
    )
    out = simplify_drc(_expr(body, ["x"])).condition

    assert isinstance(out, LogicalConnectiveNode)
    assert out.operator == "and"
    assert isinstance(out.left, NotNode)
    assert isinstance(out.right, NotNode)


def test_nnf_quantifier_flip():
    """``¬∃ x. R(x) → ∀ x. ¬R(x)``."""
    body = NotNode(
        operand=QuantifierNode(
            kind="exists",
            variables=["x"],
            body=MembershipNode(variables=["x"], relation="R"),
        ),
    )
    out = simplify_drc(_expr(body, ["dummy"])).condition

    assert isinstance(out, QuantifierNode)
    assert out.kind == "forall"
    assert isinstance(out.body, NotNode)


def test_nnf_comparison_negation():
    """``¬(< a b) → (>= a b)`` (and the symmetric flips)."""
    body = NotNode(
        operand=ComparisonNode(
            operator="<",
            left=VariableRefNode(name="a"),
            right=VariableRefNode(name="b"),
        ),
    )
    out = simplify_drc(_expr(body, ["a", "b"])).condition
    assert isinstance(out, ComparisonNode)
    assert out.operator == ">="


# ---------------------------------------------------------------------------
# Variadic and/or canonicalisation
# ---------------------------------------------------------------------------


def test_canonical_and_dedup():
    """``R(x) ∧ R(x) ∧ S(x) → R(x) ∧ S(x)`` regardless of grouping."""
    r = MembershipNode(variables=["x"], relation="R")
    s = MembershipNode(variables=["x"], relation="S")
    body = LogicalConnectiveNode(
        operator="and",
        left=LogicalConnectiveNode(operator="and", left=r, right=r),
        right=s,
    )
    out = simplify_drc(_expr(body, ["x"])).condition

    # Two unique conjuncts; the duplicate is folded.
    membs = _gather_memberships(out)
    relations = sorted(m.relation for m in membs)
    assert relations == ["R", "S"]


def test_canonical_and_or_commutativity():
    """``(R(x) ∧ S(x))`` and ``(S(x) ∧ R(x))`` produce the same tree."""
    r = MembershipNode(variables=["x"], relation="R")
    s = MembershipNode(variables=["x"], relation="S")
    a = LogicalConnectiveNode(operator="and", left=r, right=s)
    b = LogicalConnectiveNode(operator="and", left=s, right=r)

    out_a = simplify_drc(_expr(a, ["x"])).condition
    out_b = simplify_drc(_expr(b, ["x"])).condition

    # The two should be structurally identical after canonicalisation.
    from text_to_sql_planner.printer import print_lisp, PrintSuccess
    pa = print_lisp(_expr(out_a, ["x"]))
    pb = print_lisp(_expr(out_b, ["x"]))
    assert isinstance(pa, PrintSuccess)
    assert isinstance(pb, PrintSuccess)
    assert pa.output == pb.output


# ---------------------------------------------------------------------------
# Alpha-canonicalisation
# ---------------------------------------------------------------------------


def test_alpha_canonical_makes_alpha_equivalent_trees_identical():
    """Two formulas that differ only in choice of bound variable
    names should produce the same printed form after simplification."""
    # ∃ x. R(x)
    a = QuantifierNode(
        kind="exists",
        variables=["x"],
        body=MembershipNode(variables=["x"], relation="R"),
    )
    # ∃ y. R(y)
    b = QuantifierNode(
        kind="exists",
        variables=["y"],
        body=MembershipNode(variables=["y"], relation="R"),
    )

    out_a = simplify_drc(_expr(a, ["dummy"])).condition
    out_b = simplify_drc(_expr(b, ["dummy"])).condition

    from text_to_sql_planner.printer import print_lisp, PrintSuccess
    pa = print_lisp(_expr(out_a, ["dummy"]))
    pb = print_lisp(_expr(out_b, ["dummy"]))
    assert isinstance(pa, PrintSuccess)
    assert isinstance(pb, PrintSuccess)
    assert pa.output == pb.output


def test_alpha_canonical_preserves_free_variable_names():
    """Free variables (including result-variable names) must NOT be
    renamed — only bound variables get the deterministic sequence."""
    body = LogicalConnectiveNode(
        operator="and",
        left=MembershipNode(variables=["emp_id", "name"], relation="Employees"),
        right=QuantifierNode(
            kind="exists",
            variables=["x"],
            body=MembershipNode(variables=["x", "emp_id"], relation="PR"),
        ),
    )
    out = simplify_drc(_expr(body, ["emp_id", "name"])).condition

    # Every Employees membership keeps the free names ``emp_id`` and
    # ``name`` (the result variables); the inner ∃-bound name is
    # alpha-renamed.
    membs = _gather_memberships(out)
    for m in membs:
        if m.relation == "Employees":
            assert "emp_id" in m.variables
            assert "name" in m.variables
        if m.relation == "PR":
            # PR has two slots; one is the alpha-renamed bound name,
            # the other is the free ``emp_id``.
            assert "emp_id" in m.variables
