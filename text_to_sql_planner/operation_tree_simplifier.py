"""Operation-tree simplifier: tree-level rewrites that preserve the
relational meaning of a plan but produce a smaller / cleaner tree.

The planner emits a tree of relational-algebra operations. The LLM
sometimes inserts redundant intermediate steps (e.g. projecting to
``[emp_id]`` immediately before an aggregate projection ``[(COUNT
emp_id)]``). These redundancies don't affect the SQL after
:func:`text_to_sql_planner.sql.sql_converter.convert_to_sql`'s root-
projection collapsing, but they:

* clutter the human-readable operation-tree printout, and
* bloat the SQL converter's internal context-building work.

Each pass is a pure ``OperationTree → OperationTree`` rewrite that
returns the same tree object when it doesn't apply. The pipeline runs
to a fixed point.

Currently implemented passes:

1. ``_pass_redundant_inner_projection``
       ``π_C(π_{C'}(R)) → π_C(R)`` when every underlying column ``C``
       references is present in ``C'``. Aggregate projections count
       their underlying column (e.g. ``(COUNT emp_id)`` requires only
       ``emp_id``); plain-column projections need the column itself.
       This is sound under both set and bag semantics — projection
       composition: ``π_C(π_{C'}(R)) = π_C(R)`` whenever ``C ⊆ C'``.

The simplifier is opt-in via :func:`simplify_operation_tree`. The
planner doesn't run it during planning (the operation tree is the
honest record of what the LLM chose), but the SQL emission path can
run it as a post-pass to produce a cleaner final tree for printing.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable

from text_to_sql_planner.types.operation_tree import (
    OperationNode,
    OperationTree,
    OperatorNode,
    TableLeafNode,
)
from text_to_sql_planner.types.operators import ProjectionParams


# Aggregate-form recogniser for projection columns. Covers both the
# Lisp-style ``(COUNT col)`` form and the bare-call ``COUNT(col)``
# form. Lower-case variants accepted.
_AGGREGATE_FUNCS = ("COUNT", "SUM", "AVG", "MIN", "MAX")


def simplify_operation_tree(tree: OperationTree) -> OperationTree:
    """Run the simplifier pipeline to a fixed point.

    Returns a new tree (or the same tree if no rewrite applied).
    """
    if tree is None or tree.root is None:
        return tree

    current = tree.root
    for _ in range(_MAX_ITERATIONS):
        new_root = _walk_bottom_up(current, _PASSES)
        if new_root is current:
            break
        current = new_root

    if current is tree.root:
        return tree
    return OperationTree(root=current)


_MAX_ITERATIONS = 16


# ---------------------------------------------------------------------------
# Pass 1: redundant inner projection
# ---------------------------------------------------------------------------


def _pass_redundant_inner_projection(node: OperationNode) -> OperationNode:
    """``π_C(π_{C'}(R)) → π_C(R)`` when ``C ⊆ C'``.

    "C is contained in C'" here means: every column the outer
    projection's column specs reference at the underlying level is
    present in the inner projection's columns. Aggregate column specs
    contribute their underlying column (e.g. ``(COUNT emp_id)`` →
    ``emp_id``).

    When the rule fires, the outer projection's input is rewired to
    point at the inner projection's input.
    """
    if not isinstance(node, OperatorNode):
        return node
    if not isinstance(node.params, ProjectionParams):
        return node
    if not node.inputs:
        return node

    inner = node.inputs[0]
    if not isinstance(inner, OperatorNode):
        return node
    if not isinstance(inner.params, ProjectionParams):
        return node
    if not inner.inputs:
        return node

    outer_required = _columns_required_by(node.params.columns)
    inner_provided = set(_columns_provided_by(inner.params.columns))

    if not outer_required.issubset(inner_provided):
        return node

    # The inner projection is redundant. Rewire the outer projection
    # to skip it.
    return replace(node, inputs=[inner.inputs[0]])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _columns_required_by(columns: list[str]) -> set[str]:
    """Return the set of underlying columns referenced by a projection's
    column specs.

    A spec like ``(COUNT emp_id)`` or ``COUNT(emp_id)`` requires
    ``emp_id``. A bare column spec like ``emp_id`` requires
    ``emp_id``. Unrecognised forms (expressions, multi-column aggregates,
    etc.) cause this helper to refuse to simplify by returning a
    sentinel column ``__unknown__`` that won't match any real column
    name — the rule then bails out rather than risk an unsound rewrite.
    """
    out: set[str] = set()
    for spec in columns:
        underlying = _extract_underlying(spec)
        if underlying is None:
            # Refuse to simplify if we can't identify the underlying column.
            out.add("__unknown__")
        else:
            out.add(underlying)
    return out


def _columns_provided_by(columns: list[str]) -> list[str]:
    """Return the underlying-column names a projection produces in its output.

    Mirrors ``_columns_required_by`` semantics: an aggregate projection
    spec ``(COUNT emp_id)`` produces a column whose underlying name is
    ``emp_id`` (what the ``output_columns`` list of the OperatorNode
    will say). A bare column spec produces itself.

    Unrecognised forms produce ``__unknown__``, which prevents the
    outer rule from matching.
    """
    out: list[str] = []
    for spec in columns:
        underlying = _extract_underlying(spec)
        out.append(underlying if underlying is not None else "__unknown__")
    return out


_LISP_AGGREGATE_RE = re.compile(
    r"^\(\s*(?:" + "|".join(_AGGREGATE_FUNCS) + r")\s+([A-Za-z_][A-Za-z0-9_]*)\s*\)$",
    re.IGNORECASE,
)
_CALL_AGGREGATE_RE = re.compile(
    r"^(?:" + "|".join(_AGGREGATE_FUNCS) + r")\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$",
    re.IGNORECASE,
)
_BARE_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _extract_underlying(spec: str) -> str | None:
    """Return the underlying column name for a projection spec, or
    ``None`` if the spec can't be classified as a single-column
    aggregate or a bare column reference.
    """
    spec = spec.strip()

    m = _LISP_AGGREGATE_RE.match(spec)
    if m is not None:
        return m.group(1)

    m = _CALL_AGGREGATE_RE.match(spec)
    if m is not None:
        return m.group(1)

    if _BARE_COLUMN_RE.match(spec):
        return spec

    return None


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def _walk_bottom_up(
    node: OperationNode,
    passes: list[Callable[[OperationNode], OperationNode]],
) -> OperationNode:
    """Apply every pass to every operator node, bottom-up."""
    if isinstance(node, TableLeafNode):
        return node

    if isinstance(node, OperatorNode):
        new_inputs = [_walk_bottom_up(inp, passes) for inp in node.inputs]
        if any(a is not b for a, b in zip(new_inputs, node.inputs)):
            node = replace(node, inputs=new_inputs)
        for rule in passes:
            new_node = rule(node)
            if new_node is not node:
                # A rewrite fired — don't run further passes on this
                # node in the same walk; the fixed-point loop will
                # revisit if the outer caller is iterating.
                return new_node
        return node

    return node


_PASSES: list[Callable[[OperationNode], OperationNode]] = [
    _pass_redundant_inner_projection,
]
