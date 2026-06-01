"""Tests for the ``≥N − ≥(N+1) → exactly N`` pattern rewrite.

The rewrite has two pieces:

1. ``recognize_ge_n_pattern(expr)`` — pattern recogniser. Returns a
   ``GeNPattern`` describing the relation, witness slot, key slots,
   and witness names when ``expr`` encodes "≥N tuples in R sharing key
   columns", else ``None``.

2. ``emit_canonical_exactly_n(expr, pattern)`` — produces the
   canonical exactly-N DRC by appending
   ``∧ ¬∃ w_extra. R(...) ∧ ⋀_i (w_extra ≠ w_i)`` to ``expr``.

Integration test verifies that ``apply_difference(R_geN, R_geN+1)``
takes the canonical-form fast path when both inputs match the pattern.
"""

from __future__ import annotations

from text_to_sql_planner.operators.difference import apply_difference
from text_to_sql_planner.operators.exactly_n import (
    emit_canonical_exactly_n,
    recognize_ge_n_pattern,
)
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    ComparisonNode,
    DRCExpression,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)
from text_to_sql_planner.types.operators import (
    DifferenceParams,
    OperatorSuccess,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _and(*conds):
    """Left-associative AND chain."""
    if not conds:
        raise ValueError("empty and-chain")
    out = conds[0]
    for c in conds[1:]:
        out = LogicalConnectiveNode(operator="and", left=out, right=c)
    return out


def _ge_n_pattern(
    n: int,
    *,
    relation: str = "Performance_Reviews",
    arity: int = 2,
    witness_slot: int = 0,
    key_slot: int = 1,
    key_name: str = "emp_id",
    result_vars: list[str] | None = None,
) -> DRCExpression:
    """Build a ``≥N tuples in R sharing column ``key_name`` `` DRC.

    The shape is

        ∃ w1, w2, ..., wN. ⋀_i R(wi, key, *anon) ∧ ⋀_{i<j} wi ≠ wj
    """
    if result_vars is None:
        result_vars = [key_name]

    # Witnesses sorted to match the recogniser's deterministic order.
    witnesses = [f"w{i}" for i in range(1, n + 1)]

    # Build memberships.
    memberships = []
    anon_counter = [0]

    def fresh_anon() -> str:
        anon_counter[0] += 1
        return f"_a{anon_counter[0]}"

    for w in witnesses:
        slots = []
        for s in range(arity):
            if s == witness_slot:
                slots.append(w)
            elif s == key_slot:
                slots.append(key_name)
            else:
                slots.append(fresh_anon())
        memberships.append(MembershipNode(variables=slots, relation=relation))

    # Build pairwise distinctness.
    distinctness = []
    for i in range(len(witnesses)):
        for j in range(i + 1, len(witnesses)):
            distinctness.append(
                ComparisonNode(
                    operator="!=",
                    left=VariableRefNode(name=witnesses[i]),
                    right=VariableRefNode(name=witnesses[j]),
                )
            )

    body = _and(*memberships, *distinctness)

    # Bind anon names along with the witnesses.
    all_bound = list(witnesses)
    for m in memberships:
        for v in m.variables:
            if v.startswith("_a") and v not in all_bound:
                all_bound.append(v)

    return DRCExpression(
        result_variables=[ColumnVariable(name=n) for n in result_vars],
        condition=QuantifierNode(
            kind="exists", variables=all_bound, body=body,
        ),
    )


# ---------------------------------------------------------------------------
# Recogniser tests
# ---------------------------------------------------------------------------


def test_recognise_ge_2():
    expr = _ge_n_pattern(2)
    pat = recognize_ge_n_pattern(expr)
    assert pat is not None
    assert pat.relation == "Performance_Reviews"
    assert pat.witness_slot == 0
    assert pat.key_slots == {1: "emp_id"}
    assert len(pat.witnesses) == 2


def test_recognise_ge_3():
    expr = _ge_n_pattern(3)
    pat = recognize_ge_n_pattern(expr)
    assert pat is not None
    assert len(pat.witnesses) == 3


def test_recognise_rejects_no_distinctness():
    """A ≥2 candidate without the ``w1 != w2`` clause is NOT a valid ≥N pattern."""
    # Build ≥2 by hand without distinctness.
    expr = DRCExpression(
        result_variables=[ColumnVariable(name="emp_id")],
        condition=QuantifierNode(
            kind="exists",
            variables=["w1", "w2"],
            body=_and(
                MembershipNode(variables=["w1", "emp_id"], relation="R"),
                MembershipNode(variables=["w2", "emp_id"], relation="R"),
            ),
        ),
    )
    assert recognize_ge_n_pattern(expr) is None


def test_recognise_rejects_partial_distinctness():
    """``≥3`` where ``w1 != w2`` and ``w2 != w3`` but NOT ``w1 != w3`` is rejected.

    This is the "missing distinctness" failure mode that motivated the
    rewrite: a malformed ≥3 should not be picked up by the recogniser
    so the difference operator falls back to the generic encoding.
    """
    expr = DRCExpression(
        result_variables=[ColumnVariable(name="emp_id")],
        condition=QuantifierNode(
            kind="exists",
            variables=["w1", "w2", "w3"],
            body=_and(
                MembershipNode(variables=["w1", "emp_id"], relation="R"),
                MembershipNode(variables=["w2", "emp_id"], relation="R"),
                MembershipNode(variables=["w3", "emp_id"], relation="R"),
                ComparisonNode(
                    operator="!=",
                    left=VariableRefNode(name="w1"),
                    right=VariableRefNode(name="w2"),
                ),
                ComparisonNode(
                    operator="!=",
                    left=VariableRefNode(name="w2"),
                    right=VariableRefNode(name="w3"),
                ),
                # NOTE: missing (w1 != w3)
            ),
        ),
    )
    assert recognize_ge_n_pattern(expr) is None


def test_recognise_no_result_vars():
    """Empty result variables → recognition fails (nothing to be a key)."""
    expr = DRCExpression(
        result_variables=[],
        condition=MembershipNode(variables=["x"], relation="R"),
    )
    assert recognize_ge_n_pattern(expr) is None


# ---------------------------------------------------------------------------
# emit_canonical_exactly_n tests
# ---------------------------------------------------------------------------


def test_emit_appends_negated_witness():
    """``exactly N`` must add ``¬∃ w_extra. R(...) ∧ (w_extra ≠ w_i)_i``."""
    expr = _ge_n_pattern(2)
    pat = recognize_ge_n_pattern(expr)
    assert pat is not None
    out = emit_canonical_exactly_n(expr, pat)

    # Top-level: AND with the original on the left and a NotNode on
    # the right.
    assert isinstance(out.condition, LogicalConnectiveNode)
    assert out.condition.operator == "and"
    assert isinstance(out.condition.right, NotNode)

    # Inside the NotNode: ∃ w_extra. R(w_extra, emp_id) ∧ (w_extra != w1) ∧ (w_extra != w2)
    inner = out.condition.right.operand
    assert isinstance(inner, QuantifierNode)
    assert inner.kind == "exists"
    # First binder is the extra witness.
    assert "_w_extra" in inner.variables[0] or "w_extra" in inner.variables[0]


def test_emit_distinctness_against_each_witness():
    """The synthesised inner body has one ``!=`` per existing witness."""
    expr = _ge_n_pattern(3)
    pat = recognize_ge_n_pattern(expr)
    assert pat is not None
    out = emit_canonical_exactly_n(expr, pat)

    inner = out.condition.right.operand  # ∃ w_extra. body
    assert isinstance(inner, QuantifierNode)
    body = inner.body

    # Walk through the conjunction collecting distinctness operands.
    not_eq_count = 0
    stack = [body]
    while stack:
        n = stack.pop()
        if isinstance(n, ComparisonNode) and n.operator == "!=":
            not_eq_count += 1
        elif isinstance(n, LogicalConnectiveNode):
            stack.append(n.left)
            stack.append(n.right)
    assert not_eq_count == 3


# ---------------------------------------------------------------------------
# apply_difference integration tests
# ---------------------------------------------------------------------------


def test_apply_difference_takes_canonical_path_for_geN_minus_geNplus1():
    """``apply_difference(≥2, ≥3)`` should emit canonical exactly-2 form,
    not the generic ``A ∧ ¬B`` encoding.

    The canonical form is recognisable by:

    - The output condition is ``A ∧ NotNode(∃ w_extra. body)``
    - The inner ``body`` has exactly 2 distinctness comparisons
      (one per existing witness in ≥2).
    """
    ge2 = _ge_n_pattern(2)
    ge3 = _ge_n_pattern(3)
    result = apply_difference(DifferenceParams(), [ge2, ge3])
    assert isinstance(result, OperatorSuccess)

    out = result.output
    assert isinstance(out.condition, LogicalConnectiveNode)
    assert out.condition.operator == "and"
    assert isinstance(out.condition.right, NotNode)

    inner = out.condition.right.operand
    assert isinstance(inner, QuantifierNode)
    assert inner.kind == "exists"

    # Count != comparisons in the inner body.
    not_eq_count = 0
    stack = [inner.body]
    while stack:
        n = stack.pop()
        if isinstance(n, ComparisonNode) and n.operator == "!=":
            not_eq_count += 1
        elif isinstance(n, LogicalConnectiveNode):
            stack.append(n.left)
            stack.append(n.right)
    assert not_eq_count == 2


def test_apply_difference_falls_back_to_generic_for_non_geN():
    """When the inputs aren't recognisable ≥N patterns, difference
    should fall back to the generic ``C_LHS ∧ ¬C_RHS`` encoding.
    """
    # Two simple Employees-style relations — not a ≥N pattern.
    r1 = DRCExpression(
        result_variables=[ColumnVariable(name="emp_id")],
        condition=MembershipNode(variables=["emp_id"], relation="R1"),
    )
    r2 = DRCExpression(
        result_variables=[ColumnVariable(name="emp_id")],
        condition=MembershipNode(variables=["emp_id"], relation="R2"),
    )
    result = apply_difference(DifferenceParams(), [r1, r2])
    assert isinstance(result, OperatorSuccess)

    # Generic shape: AND of (membership R1) and NotNode(membership R2).
    out = result.output
    assert isinstance(out.condition, LogicalConnectiveNode)
    assert out.condition.operator == "and"
    # R1 on the left
    assert isinstance(out.condition.left, MembershipNode)
    assert out.condition.left.relation == "R1"
    # NotNode wrapping R2 on the right
    assert isinstance(out.condition.right, NotNode)
    assert isinstance(out.condition.right.operand, MembershipNode)
    assert out.condition.right.operand.relation == "R2"


def test_apply_difference_falls_back_when_witness_counts_dont_match():
    """``apply_difference(≥2, ≥4)`` is NOT a ≥N − ≥(N+1) pattern (off
    by one, witness count is 4 not 3). Falls back to generic encoding.
    """
    ge2 = _ge_n_pattern(2)
    ge4 = _ge_n_pattern(4)
    result = apply_difference(DifferenceParams(), [ge2, ge4])
    assert isinstance(result, OperatorSuccess)

    # The right side of the AND should be a NotNode wrapping the FULL
    # ≥4 condition (with all 4 witnesses bound), NOT a single
    # synthesised w_extra.
    out = result.output
    assert isinstance(out.condition, LogicalConnectiveNode)
    assert isinstance(out.condition.right, NotNode)
    inner = out.condition.right.operand
    # The original ≥4 had 4 witness binders + their anon slots → at
    # least 4 binders in the outermost quantifier of the negated body.
    assert isinstance(inner, QuantifierNode)
    assert len([v for v in inner.variables if v.startswith("w")]) == 4
