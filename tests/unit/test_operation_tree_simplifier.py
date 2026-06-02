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
