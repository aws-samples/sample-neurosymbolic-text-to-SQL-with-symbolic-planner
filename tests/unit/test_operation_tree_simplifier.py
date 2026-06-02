"""Tests for the operation-tree simplifier.

The simplifier collapses redundant intermediate projections so the
operation tree printout and SQL generator see a cleaner plan. The
critical case is the doubly-stacked projection chain the planner
emits for "How many employees ≥30 years old?" — ``π_[(COUNT emp_id)]
· π_[emp_id] · σ · T``. This test module covers that and a few
related cases (single-step, deeper chains, non-collapsible cases).
"""

from __future__ import annotations

from text_to_sql_planner.operation_tree_simplifier import simplify_operation_tree
from text_to_sql_planner.types.drc import (
    ComparisonNode,
    LiteralNode,
    VariableRefNode,
)
from text_to_sql_planner.types.operation_tree import (
    OperationTree,
    OperatorNode,
    TableLeafNode,
)
from text_to_sql_planner.types.operators import (
    ProjectionParams,
    SelectionParams,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table(name: str, columns: list[str]) -> TableLeafNode:
    return TableLeafNode(table_name=name, columns=columns)


def _projection(child, columns: list[str]) -> OperatorNode:
    return OperatorNode(
        operator="projection",
        params=ProjectionParams(columns=columns),
        inputs=[child],
        output_columns=[_underlying(c) for c in columns],
    )


def _selection(child, condition) -> OperatorNode:
    cols = (
        list(child.columns) if isinstance(child, TableLeafNode) else list(child.output_columns)
    )
    return OperatorNode(
        operator="selection",
        params=SelectionParams(condition=condition),
        inputs=[child],
        output_columns=cols,
    )


def _underlying(spec: str) -> str:
    spec = spec.strip()
    # Lisp ``(COUNT col)`` form
    if spec.startswith("(") and spec.endswith(")"):
        # ``(AGG col)`` → ``col``
        return spec[1:-1].split(None, 1)[1].strip()
    if "(" in spec and spec.endswith(")"):
        return spec.split("(", 1)[1][:-1].strip()
    return spec


def _count_projections(node) -> int:
    if isinstance(node, TableLeafNode):
        return 0
    is_proj = isinstance(node.params, ProjectionParams)
    return (1 if is_proj else 0) + sum(_count_projections(i) for i in node.inputs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_redundant_projection_collapses():
    """The nohup3.md case: ``π_[(COUNT emp_id)] · π_[emp_id] · σ · T``.

    The inner ``π_[emp_id]`` doesn't change which underlying columns
    are available to the outer aggregate projection (which only needs
    ``emp_id``), so it's redundant.
    """
    employees = _table("Employees", ["emp_id", "first_name", "date_of_birth"])
    cond = ComparisonNode(
        operator="<=",
        left=VariableRefNode(name="date_of_birth"),
        right=LiteralNode(value=20000, data_type="number"),
    )
    sel = _selection(employees, cond)
    proj_inner = _projection(sel, ["emp_id"])
    proj_outer = _projection(proj_inner, ["(COUNT emp_id)"])
    tree = OperationTree(root=proj_outer)

    simplified = simplify_operation_tree(tree)

    # Only one projection should remain (the outer aggregate).
    assert _count_projections(simplified.root) == 1
    # The remaining projection's input is the selection (no longer the
    # inner projection).
    assert isinstance(simplified.root, OperatorNode)
    assert isinstance(simplified.root.params, ProjectionParams)
    assert simplified.root.params.columns == ["(COUNT emp_id)"]
    assert simplified.root.inputs[0] is sel


def test_chain_of_three_projections_collapses():
    """``π_C1 · π_C2 · π_C3 · T`` collapses to ``π_C1 · T`` when each
    outer step's required columns are provided by the next."""
    table = _table("T", ["a", "b", "c"])
    p3 = _projection(table, ["a", "b"])
    p2 = _projection(p3, ["a"])
    p1 = _projection(p2, ["a"])
    tree = OperationTree(root=p1)

    simplified = simplify_operation_tree(tree)

    # Down to one projection over the table.
    assert _count_projections(simplified.root) == 1
    # The single remaining projection sits directly on top of T.
    assert isinstance(simplified.root.inputs[0], TableLeafNode)


def test_aggregate_form_recognised():
    """Both ``(COUNT col)`` and ``COUNT(col)`` forms collapse the
    inner ``π_[col]`` step."""
    for outer_spec in ["(COUNT emp_id)", "COUNT(emp_id)"]:
        table = _table("Employees", ["emp_id", "first_name"])
        inner = _projection(table, ["emp_id"])
        outer = _projection(inner, [outer_spec])
        tree = OperationTree(root=outer)

        simplified = simplify_operation_tree(tree)

        assert _count_projections(simplified.root) == 1, (
            f"failed for spec {outer_spec!r}"
        )
        assert isinstance(simplified.root.inputs[0], TableLeafNode)


def test_non_redundant_projection_kept():
    """``π_[a] · π_[a, b] · T`` — outer needs only ``a`` but the
    inner projection narrows away ``b``. Inner is required if any
    later step needs ``b``, but as a leaf rewrite (the outer is the
    final consumer), inner *is* redundant under set semantics. The
    rule fires.

    Conversely, a non-redundant case is when the outer references a
    column NOT in the inner's output: ``π_[c] · π_[a, b] · T`` — the
    rule must NOT fire (and indeed our recogniser refuses because
    ``c ∉ inner_provided``).
    """
    table = _table("T", ["a", "b", "c"])
    inner = _projection(table, ["a", "b"])  # provides {a, b}
    outer = _projection(inner, ["c"])  # needs {c} which is not in {a, b}
    tree = OperationTree(root=outer)

    simplified = simplify_operation_tree(tree)

    # Both projections should still be present.
    assert _count_projections(simplified.root) == 2


def test_unknown_form_blocks_simplification():
    """If the outer projection contains an unrecognised form (e.g. an
    expression like ``a + b``), the rule must NOT fire — we can't
    safely tell which underlying columns are needed.
    """
    table = _table("T", ["a", "b"])
    inner = _projection(table, ["a", "b"])
    outer = _projection(inner, ["a + b"])  # unknown form
    tree = OperationTree(root=outer)

    simplified = simplify_operation_tree(tree)

    # Both projections kept.
    assert _count_projections(simplified.root) == 2


def test_simplifier_idempotent():
    """Running the simplifier twice produces the same tree as once."""
    table = _table("T", ["a", "b", "c"])
    p3 = _projection(table, ["a", "b"])
    p2 = _projection(p3, ["a"])
    p1 = _projection(p2, ["a"])
    tree = OperationTree(root=p1)

    once = simplify_operation_tree(tree)
    twice = simplify_operation_tree(once)

    # Same tree shape.
    assert _count_projections(once.root) == _count_projections(twice.root)


def test_no_projection_no_change():
    """A tree without any projection chain is returned as-is."""
    table = _table("T", ["a"])
    tree = OperationTree(root=table)

    simplified = simplify_operation_tree(tree)
    assert simplified.root is table


def test_simplifier_handles_none():
    """``None`` tree / root is returned without crashing."""
    assert simplify_operation_tree(None) is None
    empty = OperationTree(root=None)
    assert simplify_operation_tree(empty) is empty



# ---------------------------------------------------------------------------
# Pass: Join over Difference → AntiJoin
# ---------------------------------------------------------------------------

from text_to_sql_planner.types.operators import (
    AntiJoinParams,
    DifferenceParams,
    JoinParams,
)


def _join(left, right, key_cols: list[str], output_cols: list[str]) -> OperatorNode:
    return OperatorNode(
        operator="join",
        params=JoinParams(join_columns=key_cols),
        inputs=[left, right],
        output_columns=output_cols,
    )


def _difference(left, right, output_cols: list[str]) -> OperatorNode:
    return OperatorNode(
        operator="difference",
        params=DifferenceParams(),
        inputs=[left, right],
        output_columns=output_cols,
    )


def test_join_difference_collapses_to_antijoin():
    """``Join(L, Difference(L', R), key=k) → AntiJoin(L, R, key=k)``
    when ``L'`` is a projection chain over ``L``.
    """
    employees = _table("Employees", ["emp_id", "first_name", "last_name"])
    reviews = _table("Performance_Reviews", ["review_id", "emp_id"])

    emp_proj = _projection(employees, ["emp_id"])
    rev_proj = _projection(reviews, ["emp_id"])
    diff = _difference(emp_proj, rev_proj, ["emp_id"])
    join = _join(employees, diff, ["emp_id"], ["emp_id", "first_name", "last_name"])

    tree = OperationTree(root=join)
    simplified = simplify_operation_tree(tree)

    assert isinstance(simplified.root, OperatorNode)
    assert isinstance(simplified.root.params, AntiJoinParams)
    assert simplified.root.params.join_columns == ["emp_id"]
    # Left is the original Employees table.
    assert isinstance(simplified.root.inputs[0], TableLeafNode)
    assert simplified.root.inputs[0].table_name == "Employees"
    # Right input has the projection chain stripped down to the base.
    assert isinstance(simplified.root.inputs[1], TableLeafNode)
    assert simplified.root.inputs[1].table_name == "Performance_Reviews"


def test_join_difference_with_direct_left_collapses():
    """``Join(L, Difference(L, R), key=k) → AntiJoin(L, R, key=k)`` when
    the difference's left input is the same table directly (no
    intermediate projection)."""
    t1 = _table("T1", ["a", "b"])
    t2 = _table("T2", ["a"])
    diff = _difference(t1, t2, ["a", "b"])
    join = _join(t1, diff, ["a"], ["a", "b"])

    tree = OperationTree(root=join)
    simplified = simplify_operation_tree(tree)

    assert isinstance(simplified.root.params, AntiJoinParams)


def test_join_difference_does_not_collapse_when_left_differs():
    """If the join's left and the diff's left are different physical
    relations, the simple ``Join(L, Difference(L, R)) → AntiJoin``
    pass doesn't fire. Instead, the three-way pass kicks in, producing
    ``AntiJoin(Join(L, M), R)``.
    """
    t1 = _table("T1", ["a"])
    t2 = _table("T2", ["a"])
    t3 = _table("T3", ["a"])
    diff = _difference(t2, t3, ["a"])
    join = _join(t1, diff, ["a"], ["a"])

    tree = OperationTree(root=join)
    simplified = simplify_operation_tree(tree)

    # The three-way pass produces an AntiJoin whose left input is
    # itself a Join(T1, T2).
    assert isinstance(simplified.root.params, AntiJoinParams)
    inner = simplified.root.inputs[0]
    assert isinstance(inner, OperatorNode)
    assert isinstance(inner.params, JoinParams)
    assert isinstance(inner.inputs[0], TableLeafNode)
    assert inner.inputs[0].table_name == "T1"
    assert isinstance(inner.inputs[1], TableLeafNode)
    assert inner.inputs[1].table_name == "T2"
    # The AntiJoin's right input is T3.
    assert isinstance(simplified.root.inputs[1], TableLeafNode)
    assert simplified.root.inputs[1].table_name == "T3"


def test_join_difference_does_not_collapse_when_right_is_not_difference():
    """The right input must be a Difference operator. Anything else
    (selection, join, projection, etc.) doesn't qualify."""
    t1 = _table("T1", ["a"])
    t2 = _table("T2", ["a"])
    proj = _projection(t2, ["a"])  # right is a projection, not a difference
    join = _join(t1, proj, ["a"], ["a"])

    tree = OperationTree(root=join)
    simplified = simplify_operation_tree(tree)

    # Join preserved.
    assert isinstance(simplified.root.params, JoinParams)


def test_join_difference_with_selection_inside_left_collapses_via_three_way():
    """``Join(T1, Difference(σ_p(T1), T2))`` — the simple
    ``Join(L, Difference(L, R)) → AntiJoin`` pass refuses to fire
    (selection on the diff's left makes the row sets unequal). But
    the three-way pass DOES fire: it treats σ_p(T1) as ``M``,
    producing ``AntiJoin(Join(T1, σ_p(T1)), T2)``. Under set
    semantics this is sound.
    """
    t1 = _table("Employees", ["emp_id"])
    t2 = _table("Reviews", ["emp_id"])
    cond = ComparisonNode(
        operator="=",
        left=VariableRefNode(name="emp_id"),
        right=LiteralNode(value=1, data_type="number"),
    )
    sel = _selection(t1, cond)
    diff = _difference(sel, t2, ["emp_id"])
    join = _join(t1, diff, ["emp_id"], ["emp_id"])

    tree = OperationTree(root=join)
    simplified = simplify_operation_tree(tree)

    # Three-way pass produces AntiJoin(Join(T1, σ_p(T1)), T2).
    assert isinstance(simplified.root.params, AntiJoinParams)



# ---------------------------------------------------------------------------
# Pass: Difference of L and Join(L, R) → AntiJoin
# ---------------------------------------------------------------------------


def test_difference_join_collapses_to_antijoin():
    """``Difference(L, Join(L, R, k)) → AntiJoin(L, R, k)`` — the
    common ``L − (L ⋈ R)`` shape the planner emits."""
    employees = _table("Employees", ["emp_id", "first_name", "last_name"])
    reviews = _table("Performance_Reviews", ["review_id", "emp_id"])

    join = _join(
        employees,
        reviews,
        ["emp_id"],
        ["emp_id", "first_name", "last_name", "review_id"],
    )
    join_proj = _projection(join, ["emp_id", "first_name", "last_name"])
    diff = _difference(
        employees,
        join_proj,
        ["emp_id", "first_name", "last_name"],
    )
    tree = OperationTree(root=diff)

    simplified = simplify_operation_tree(tree)

    assert isinstance(simplified.root, OperatorNode)
    assert isinstance(simplified.root.params, AntiJoinParams)
    assert simplified.root.params.join_columns == ["emp_id"]
    # Left input is the original Employees table.
    assert isinstance(simplified.root.inputs[0], TableLeafNode)
    assert simplified.root.inputs[0].table_name == "Employees"
    # Right input is the Performance_Reviews table directly (no
    # projection wrapping).
    assert isinstance(simplified.root.inputs[1], TableLeafNode)
    assert simplified.root.inputs[1].table_name == "Performance_Reviews"


def test_difference_join_no_projection_on_join_collapses():
    """When the inner join's column shape matches L directly (no
    projection), the rewrite still fires."""
    t1 = _table("T1", ["a", "b"])
    t2 = _table("T2", ["a"])
    join = _join(t1, t2, ["a"], ["a", "b"])
    diff = _difference(t1, join, ["a", "b"])
    tree = OperationTree(root=diff)

    simplified = simplify_operation_tree(tree)

    assert isinstance(simplified.root.params, AntiJoinParams)


def test_difference_join_does_not_collapse_when_left_differs():
    """Diff's left and join's left must be the same physical relation."""
    t1 = _table("T1", ["a"])
    t2 = _table("T2", ["a"])
    t3 = _table("T3", ["a"])
    # Inner join is T2 ⋈ T3 (not T1 ⋈ anything).
    join = _join(t2, t3, ["a"], ["a"])
    diff = _difference(t1, join, ["a"])
    tree = OperationTree(root=diff)

    simplified = simplify_operation_tree(tree)

    # Difference preserved (no rewrite).
    assert isinstance(simplified.root.params, DifferenceParams)


def test_difference_join_does_not_collapse_when_right_is_not_join():
    """The diff's right input (after stripping projections) must be a
    Join. A bare table or a Difference doesn't qualify."""
    t1 = _table("T1", ["a"])
    t2 = _table("T2", ["a"])
    # Right input is just T2 (no inner join).
    diff = _difference(t1, t2, ["a"])
    tree = OperationTree(root=diff)

    simplified = simplify_operation_tree(tree)
    assert isinstance(simplified.root.params, DifferenceParams)


def test_difference_join_with_selection_inside_join_left_does_not_collapse():
    """``L − Join(σ_p(L), R)`` ≠ ``L ▷ R`` — the selection narrows the
    join's first input, so the difference would still keep rows of
    ``L`` that satisfy ``¬p`` (regardless of whether they have an R
    match). The rewrite must NOT fire when the inner join's left has
    a selection on top of the matching base.
    """
    t1 = _table("Employees", ["emp_id"])
    t2 = _table("Reviews", ["emp_id"])
    cond = ComparisonNode(
        operator="=",
        left=VariableRefNode(name="emp_id"),
        right=LiteralNode(value=1, data_type="number"),
    )
    sel = _selection(t1, cond)
    join = _join(sel, t2, ["emp_id"], ["emp_id"])
    diff = _difference(t1, join, ["emp_id"])
    tree = OperationTree(root=diff)

    simplified = simplify_operation_tree(tree)
    assert isinstance(simplified.root.params, DifferenceParams)



# ---------------------------------------------------------------------------
# Pass: Three-way Join over Difference → AntiJoin(Join(L, M), R)
# ---------------------------------------------------------------------------


def test_three_way_join_difference_collapses():
    """The user's three-way example: ``Join(L, Difference(M, R))``
    where L, M, R are three distinct tables.

    Expected: ``AntiJoin(Join(L, M), R)`` — semi-join with M plus
    anti-join with R.
    """
    employees = _table(
        "Employees", ["emp_id", "first_name", "last_name"]
    )
    training = _table(
        "Training_Enrollment", ["enrollment_id", "emp_id"]
    )
    reviews = _table("Performance_Reviews", ["review_id", "emp_id"])

    training_proj = _projection(training, ["emp_id"])
    reviews_proj = _projection(reviews, ["emp_id"])
    diff = _difference(training_proj, reviews_proj, ["emp_id"])
    join = _join(
        employees,
        diff,
        ["emp_id"],
        ["emp_id", "first_name", "last_name"],
    )

    tree = OperationTree(root=join)
    simplified = simplify_operation_tree(tree)

    assert isinstance(simplified.root, OperatorNode)
    assert isinstance(simplified.root.params, AntiJoinParams)
    # Inner is Join(Employees, Training_Enrollment).
    inner = simplified.root.inputs[0]
    assert isinstance(inner, OperatorNode)
    assert isinstance(inner.params, JoinParams)
    assert isinstance(inner.inputs[0], TableLeafNode)
    assert inner.inputs[0].table_name == "Employees"
    assert isinstance(inner.inputs[1], TableLeafNode)
    assert inner.inputs[1].table_name == "Training_Enrollment"
    # AntiJoin's right is Performance_Reviews.
    assert isinstance(simplified.root.inputs[1], TableLeafNode)
    assert simplified.root.inputs[1].table_name == "Performance_Reviews"


def test_three_way_pass_does_not_fire_when_M_equals_L():
    """When M = L, the simpler Join(L, Difference(L, R)) → AntiJoin(L, R)
    pass should win. The three-way pass bails to avoid producing a
    redundant ``AntiJoin(Join(L, L), R)`` shape.
    """
    employees = _table("Employees", ["emp_id"])
    reviews = _table("Performance_Reviews", ["emp_id"])

    diff = _difference(employees, reviews, ["emp_id"])
    join = _join(employees, diff, ["emp_id"], ["emp_id"])

    tree = OperationTree(root=join)
    simplified = simplify_operation_tree(tree)

    # Result is a clean AntiJoin(Employees, Performance_Reviews) — no
    # redundant inner Join(Employees, Employees).
    assert isinstance(simplified.root.params, AntiJoinParams)
    assert isinstance(simplified.root.inputs[0], TableLeafNode)
    assert simplified.root.inputs[0].table_name == "Employees"
    assert isinstance(simplified.root.inputs[1], TableLeafNode)
    assert simplified.root.inputs[1].table_name == "Performance_Reviews"



# ---------------------------------------------------------------------------
# Pass: drop_redundant_filtering_join
# ---------------------------------------------------------------------------
#
# ``Join(L, π_cols(Join(A, B, k)), k) → Join(L, π_cols(A), k)`` when
# ``L.k`` is provably ``⊆ B.k``. The role of the inner ``A ⋈ B`` was
# to filter A to rows whose key has a B-match; when L's key is
# already known to be in B (by construction), the inner B-filter is
# redundant.

from text_to_sql_planner.types.operators import (
    CartesianProductParams,
)


def test_drop_redundant_filtering_join_basic():
    """The nohup3.md scenario: outer join's left is a self-join of
    Performance_Reviews (so its emp_id is provably ⊆ PR.emp_id), and
    the right is ``π_{emp,fname,lname}(Employees ⋈ PR)``. The inner
    ``Employees ⋈ PR`` is redundant — drop it.
    """
    pr = _table("Performance_Reviews", ["review_id", "emp_id"])
    emp = _table("Employees", ["emp_id", "first_name", "last_name"])

    # L = a per-row relation backed entirely by PR (a simple
    # projection here; the real planner emits a multi-witness self-
    # join, but the provenance check works on either).
    L = _projection(pr, ["emp_id"])

    # R = π_{emp_id, first_name, last_name}(Employees ⋈ PR on emp_id)
    emp_pr_join = OperatorNode(
        operator="join",
        params=JoinParams(join_columns=["emp_id"]),
        inputs=[emp, pr],
        output_columns=[
            "emp_id", "first_name", "last_name", "review_id",
        ],
    )
    R = _projection(
        emp_pr_join, ["emp_id", "first_name", "last_name"],
    )

    outer_join = _join(
        L, R, ["emp_id"], ["emp_id", "first_name", "last_name"],
    )
    tree = OperationTree(root=outer_join)

    simplified = simplify_operation_tree(tree)

    # The outer Join survives but its right input is now a projection
    # of Employees alone (no inner ⋈ PR).
    assert isinstance(simplified.root, OperatorNode)
    assert isinstance(simplified.root.params, JoinParams)
    new_right = simplified.root.inputs[1]
    assert isinstance(new_right, OperatorNode)
    assert isinstance(new_right.params, ProjectionParams)
    # The projection's input is now the Employees table directly.
    assert isinstance(new_right.inputs[0], TableLeafNode)
    assert new_right.inputs[0].table_name == "Employees"


def test_drop_redundant_filtering_join_through_self_join():
    """L is a multi-witness Performance_Reviews self-join (the
    "≥3 reviews" pattern). The provenance check still concludes
    ``L.emp_id ⊆ PR.emp_id``, so the rule fires.
    """
    pr = _table("Performance_Reviews", ["review_id", "emp_id"])
    emp = _table("Employees", ["emp_id", "first_name", "last_name"])

    p1 = _projection(pr, ["review_id", "emp_id"])
    p2 = _projection(pr, ["review_id", "emp_id"])
    p3 = _projection(pr, ["review_id", "emp_id"])
    cp1 = OperatorNode(
        operator="cartesian_product",
        params=CartesianProductParams(),
        inputs=[p1, p2],
        output_columns=["review_id_1", "emp_id_1", "review_id_2", "emp_id_2"],
    )
    cp2 = OperatorNode(
        operator="cartesian_product",
        params=CartesianProductParams(),
        inputs=[cp1, p3],
        output_columns=[
            "review_id_1", "emp_id_1",
            "review_id_2", "emp_id_2",
            "review_id", "emp_id",
        ],
    )
    L = _projection(cp2, ["emp_id"])

    emp_pr_join = OperatorNode(
        operator="join",
        params=JoinParams(join_columns=["emp_id"]),
        inputs=[emp, pr],
        output_columns=[
            "emp_id", "first_name", "last_name", "review_id",
        ],
    )
    R = _projection(
        emp_pr_join, ["emp_id", "first_name", "last_name"],
    )
    outer_join = _join(
        L, R, ["emp_id"], ["emp_id", "first_name", "last_name"],
    )
    tree = OperationTree(root=outer_join)

    simplified = simplify_operation_tree(tree)

    new_right = simplified.root.inputs[1]
    assert isinstance(new_right, OperatorNode)
    assert isinstance(new_right.params, ProjectionParams)
    assert isinstance(new_right.inputs[0], TableLeafNode)
    assert new_right.inputs[0].table_name == "Employees"


def test_does_not_fire_when_L_is_not_backed_by_B():
    """L is built from Departments (not Performance_Reviews). The
    inner ``Employees ⋈ Performance_Reviews`` filter is NOT redundant
    — dropping it would change the result by including employees who
    have no performance review.
    """
    dept = _table("Departments", ["dept_id", "emp_id"])
    pr = _table("Performance_Reviews", ["review_id", "emp_id"])
    emp = _table("Employees", ["emp_id", "first_name", "last_name"])

    L = _projection(dept, ["emp_id"])
    emp_pr_join = OperatorNode(
        operator="join",
        params=JoinParams(join_columns=["emp_id"]),
        inputs=[emp, pr],
        output_columns=["emp_id", "first_name", "last_name", "review_id"],
    )
    R = _projection(emp_pr_join, ["emp_id", "first_name", "last_name"])
    outer_join = _join(
        L, R, ["emp_id"], ["emp_id", "first_name", "last_name"],
    )
    tree = OperationTree(root=outer_join)

    simplified = simplify_operation_tree(tree)

    # Outer join still has the inner ``Employees ⋈ PR`` on its right.
    new_right = simplified.root.inputs[1]
    # Walk through any projection wrapper.
    while (
        isinstance(new_right, OperatorNode)
        and isinstance(new_right.params, ProjectionParams)
    ):
        new_right = new_right.inputs[0]
    assert isinstance(new_right, OperatorNode)
    assert isinstance(new_right.params, JoinParams)


def test_does_not_fire_when_inner_join_key_differs():
    """The inner join's key must match the outer join's. If the inner
    is on a different key, the redundancy argument doesn't apply.
    """
    pr = _table("Performance_Reviews", ["review_id", "emp_id"])
    emp = _table("Employees", ["emp_id", "first_name", "last_name", "dept_id"])
    dept = _table("Departments", ["dept_id", "dept_name"])

    L = _projection(pr, ["emp_id"])

    # Inner join on dept_id (not emp_id).
    emp_dept_join = OperatorNode(
        operator="join",
        params=JoinParams(join_columns=["dept_id"]),
        inputs=[emp, dept],
        output_columns=[
            "emp_id", "first_name", "last_name", "dept_id", "dept_name",
        ],
    )
    R = _projection(
        emp_dept_join, ["emp_id", "first_name", "last_name"],
    )

    outer_join = _join(
        L, R, ["emp_id"], ["emp_id", "first_name", "last_name"],
    )
    tree = OperationTree(root=outer_join)

    simplified = simplify_operation_tree(tree)

    new_right = simplified.root.inputs[1]
    while (
        isinstance(new_right, OperatorNode)
        and isinstance(new_right.params, ProjectionParams)
    ):
        new_right = new_right.inputs[0]
    # Inner join preserved (different key).
    assert isinstance(new_right, OperatorNode)
    assert isinstance(new_right.params, JoinParams)


def test_no_projection_wrapper_on_right_works_too():
    """Right input is the inner ``Join(A, B, k)`` directly, with no
    surrounding projection. The rule still fires; the rewrite drops
    B and uses A directly.
    """
    pr = _table("Performance_Reviews", ["review_id", "emp_id"])
    emp = _table("Employees", ["emp_id", "first_name", "last_name"])

    L = _projection(pr, ["emp_id"])
    R = OperatorNode(
        operator="join",
        params=JoinParams(join_columns=["emp_id"]),
        inputs=[emp, pr],
        output_columns=[
            "emp_id", "first_name", "last_name", "review_id",
        ],
    )
    outer_join = _join(
        L, R, ["emp_id"], ["emp_id", "first_name", "last_name"],
    )
    tree = OperationTree(root=outer_join)

    simplified = simplify_operation_tree(tree)

    new_right = simplified.root.inputs[1]
    assert isinstance(new_right, TableLeafNode)
    assert new_right.table_name == "Employees"
