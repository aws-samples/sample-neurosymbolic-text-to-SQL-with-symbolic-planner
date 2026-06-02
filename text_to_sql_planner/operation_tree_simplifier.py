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

2. ``_pass_join_difference_to_antijoin``
       ``Join(L, Difference(X, R), key=k) → AntiJoin(L, R, key=k)``
       when ``X`` is the same physical relation as ``L`` (modulo a
       projection chain). The join filters ``L`` to rows whose key is
       in ``(X.k − R.k)``; when ``X.k = L.k``, that's exactly the
       anti-join ``L ▷ R``. The SQL converter renders the rewritten
       form as a ``NOT EXISTS`` correlated subquery, which is both
       more readable and easier on the query optimiser than the
       original ``JOIN (SELECT * FROM (… EXCEPT …))`` shape.

3. ``_pass_difference_join_to_antijoin``
       ``Difference(L, Join(X, R, key=k)) → AntiJoin(L, R, key=k)``
       when ``X`` is the same physical relation as ``L`` (again
       modulo a projection chain). ``Join(L, R, k)`` produces
       exactly the rows of ``L`` whose key has at least one match
       in ``R``; subtracting that from ``L`` yields the rows with
       NO match — the anti-join ``L ▷ R``. Like the previous pass,
       the rewrite emits as a ``NOT EXISTS`` correlated subquery,
       collapsing what the planner expressed as ``L − (L ⋈ R)``
       into the natural anti-join shape.

4. ``_pass_join_three_way_difference_to_antijoin``
       ``Join(L, Difference(M, R), key=k) → AntiJoin(Join(L, M, k), R, k)``
       where ``M`` is a *different* relation from ``L``. Generalises
       pass 2 to the case where the difference's left input ``M`` is
       not the same as the join's left ``L`` (e.g. "employees enrolled
       in training but with no performance reviews":
       ``Join(Employees, Training − Reviews, emp_id)``). The
       decomposition is unconditionally sound:
       ``l.k ∈ (M.k − R.k) ≡ (l.k ∈ M.k) ∧ (l.k ∉ R.k)``. The output
       renders as ``L ⋈ M ON L.k = M.k WHERE NOT EXISTS (… R …
       WHERE R.k = L.k)``. Set semantics from the original
       ``Difference`` are recovered by the SQL converter's
       ``DISTINCT`` flag — the planner's caller is responsible for
       setting ``distinct=True`` when the question requires set
       semantics.

5. ``_pass_drop_redundant_filtering_join``
       ``Join(L, π_cols(Join(A, B, k)), k) → Join(L, A, k)`` when
       ``L.k ⊆ B.k`` is provably true by tree provenance. The inner
       ``Join(A, B, k)`` was filtering ``A`` to rows whose key has a
       match in ``B``; when ``L.k ⊆ B.k`` (i.e. every L key is
       already a B key by tree construction), the outer ``Join(L,
       …, k)`` would re-filter to ``L.k ∩ A.k ∩ B.k = L.k ∩ A.k``,
       so the inner B-filter is redundant. This collapses the planner's
       common "join-with-an-Employees⋈Reviews-decoration" pattern
       (used to attach ``first_name``/``last_name`` to a per-row
       relation already known to be backed by Reviews) into a clean
       ``Join(L, A, k)``.

4. ``_pass_join_three_way_difference_to_antijoin``
       ``Join(L, Difference(M, R), key=k) → AntiJoin(Join(L, M, k), R, k)``
       for the *three-relation* case where ``L``, ``M``, ``R`` are
       distinct relations (the existing
       ``_pass_join_difference_to_antijoin`` handles ``M = L``).
       Decomposes ``L ⋈ (M − R)`` into a semi-join with ``M``
       (rendered as a regular ``JOIN``) plus an anti-join against
       ``R`` (rendered as ``NOT EXISTS``). Because the outer
       ``EXCEPT`` was set-based, the rewrite needs ``SELECT
       DISTINCT`` at the top level to preserve cardinality —
       handled by the SQL converter's existing ``distinct`` flag
       at finalisation.

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
from text_to_sql_planner.types.operators import (
    AntiJoinParams,
    CartesianProductParams,
    DifferenceParams,
    JoinParams,
    ProjectionParams,
    SelectionParams,
    UnionParams,
)


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
# Pass 2: Join over Difference → AntiJoin
# ---------------------------------------------------------------------------


def _pass_join_difference_to_antijoin(node: OperationNode) -> OperationNode:
    """``Join(L, Difference(X, R), key=k) → AntiJoin(L, R, key=k)``
    when ``X`` is the same physical relation as ``L`` (modulo a chain
    of plain projections that preserve the join key columns).

    Soundness:

    * ``Join(L, X − R)`` on ``k`` produces ``{l ∈ L | l.k ∈ (X.k − R.k)}``
      (treating the join as a filter on ``L`` since the difference's
      output equals ``L``'s key projection).
    * When ``X`` is ``L`` itself or a projection chain over ``L``, we
      have ``X.k ⊆ L.k``. Combined with ``l ∈ L``, the constraint
      ``l.k ∈ X.k`` is automatic, so the filter reduces to
      ``l.k ∉ R.k``, i.e. the anti-join ``L ▷ R`` on ``k``.
    * The conservative "same physical relation" check: ``X`` and ``L``,
      after stripping any pure-projection chain, must be the same
      ``TableLeafNode`` by table name. We don't recurse through
      selections, joins, etc. — those would change the row set, and
      the equivalence ``X.k ⊆ L.k`` would no longer hold.

    The check uses table-name equality (with column-list equality as
    a tiebreak) rather than object identity because the planner clones
    nodes during operator application.

    The output ``output_columns`` of the new ``AntiJoinNode`` is L's
    column shape — anti-join is a filter on L, not a column-merging
    operator.
    """
    if not isinstance(node, OperatorNode):
        return node
    if not isinstance(node.params, JoinParams):
        return node
    if len(node.inputs) != 2:
        return node
    if not node.params.join_columns:
        return node

    left, right = node.inputs[0], node.inputs[1]

    # Right input must be a Difference operator.
    if not isinstance(right, OperatorNode):
        return node
    if not isinstance(right.params, DifferenceParams):
        return node
    if len(right.inputs) != 2:
        return node

    diff_left, diff_right = right.inputs[0], right.inputs[1]

    # Walk through any chain of projections on diff_left to its base
    # relation. A "projection chain" is a sequence of ProjectionParams
    # operators each applied to a single input.
    diff_left_base = _strip_projection_chain(diff_left)

    if not _same_physical_relation(left, diff_left_base):
        return node

    # Strip any projection chain from the difference's right input
    # too: the anti-join's RHS only needs to expose the join keys, so
    # we can use the underlying base relation directly. This makes the
    # NOT EXISTS subquery reference the table directly instead of a
    # wrapped ``SELECT k FROM T`` form.
    diff_right_base = _strip_projection_chain(diff_right)

    # All preconditions satisfied — rewrite to AntiJoin.
    return OperatorNode(
        operator="anti_join",
        params=AntiJoinParams(join_columns=list(node.params.join_columns)),
        inputs=[left, diff_right_base],
        output_columns=list(node.output_columns),
        output_expression=node.output_expression,
    )


def _strip_projection_chain(node: OperationNode) -> OperationNode:
    """Walk through any chain of ``ProjectionParams`` operators and
    return the first non-projection node beneath them.

    A projection over ``R`` produces a relation whose tuples are a
    column-narrowed view of ``R``'s; the row set (in terms of the
    join-key columns we care about) is unchanged so long as the
    projection retains those columns. The caller has already verified
    via the difference-output column shape that the keys are present.
    """
    current = node
    while (
        isinstance(current, OperatorNode)
        and isinstance(current.params, ProjectionParams)
        and current.inputs
    ):
        current = current.inputs[0]
    return current


def _same_physical_relation(a: OperationNode, b: OperationNode) -> bool:
    """Return True iff ``a`` and ``b`` are the same base table.

    Both must be ``TableLeafNode`` instances with equal table names
    and equal column lists. The table-name check is the load-bearing
    invariant; the column-list check is a defensive tiebreak that
    catches the (rare) case where the planner introduces two
    differently-shaped views of the same table.
    """
    if not isinstance(a, TableLeafNode) or not isinstance(b, TableLeafNode):
        return False
    if a.table_name != b.table_name:
        return False
    return a.columns == b.columns


# ---------------------------------------------------------------------------
# Pass 3: Difference of L and Join(L, R) → AntiJoin
# ---------------------------------------------------------------------------


def _pass_difference_join_to_antijoin(node: OperationNode) -> OperationNode:
    """``Difference(L, Join(X, R, key=k)) → AntiJoin(L, R, key=k)``
    when ``X`` is the same physical relation as ``L`` (modulo a chain
    of plain projections that preserve the join key columns).

    Soundness:

    * ``Join(L, R, k)`` produces ``{l ∈ L | ∃ r ∈ R. l.k = r.k}`` —
      i.e. rows of ``L`` whose key has at least one match in ``R``.
      The output column shape is ``L``'s columns extended with ``R``'s
      non-key columns. For the difference to be type-compatible with
      ``L``, the join's output column shape must equal ``L``'s — which
      means either the join projects only ``L``'s columns, or the
      planner inserts a projection back to ``L``'s shape on top of
      the join. We strip any such projection chain on the join.
    * ``Difference(L, Join(L, R, k))`` then keeps rows of ``L`` whose
      tuple is *not* in the join's output. A tuple is in the join's
      output iff it's in ``L`` (trivially, by the join's projection)
      AND has a matching ``R`` row on ``k``. So the difference keeps
      rows of ``L`` with no matching ``R`` row → exactly the
      anti-join ``L ▷ R`` on ``k``.
    * Conservatism: as with the join-difference pass, ``X`` must be
      the same physical relation as ``L`` (after stripping pure
      projections). A selection or further join on the diff's
      ``X`` side would produce a strict subset, breaking the
      ``X.k ⊆ L.k`` precondition the soundness argument relies on.

    The output ``output_columns`` of the new ``AntiJoinNode`` is the
    Difference's column shape, which equals ``L``'s column shape.
    """
    if not isinstance(node, OperatorNode):
        return node
    if not isinstance(node.params, DifferenceParams):
        return node
    if len(node.inputs) != 2:
        return node

    left, right = node.inputs[0], node.inputs[1]

    # The right input is the inner Join. The planner often layers a
    # projection on top of the join to bring its column shape back
    # down to match the difference's left side; strip that chain.
    right_stripped = _strip_projection_chain(right)
    if not isinstance(right_stripped, OperatorNode):
        return node
    if not isinstance(right_stripped.params, JoinParams):
        return node
    if len(right_stripped.inputs) != 2:
        return node
    if not right_stripped.params.join_columns:
        return node

    inner_left, inner_right = right_stripped.inputs[0], right_stripped.inputs[1]

    # Walk through any projection chain on the inner join's left to
    # its base. If that base matches the difference's left base, the
    # rewrite is sound.
    inner_left_base = _strip_projection_chain(inner_left)
    left_base = _strip_projection_chain(left)

    if not _same_physical_relation(left_base, inner_left_base):
        return node

    # Strip any projection chain from the inner join's right input
    # too: the anti-join's RHS only needs the join keys, so we can
    # use the underlying base relation directly.
    inner_right_base = _strip_projection_chain(inner_right)

    return OperatorNode(
        operator="anti_join",
        params=AntiJoinParams(
            join_columns=list(right_stripped.params.join_columns),
        ),
        inputs=[left, inner_right_base],
        output_columns=list(node.output_columns),
        output_expression=node.output_expression,
    )


# ---------------------------------------------------------------------------
# Pass 4: Three-way Join over Difference → AntiJoin(Join(L, M), R)
# ---------------------------------------------------------------------------


def _pass_join_three_way_difference_to_antijoin(node: OperationNode) -> OperationNode:
    """``Join(L, Difference(M, R), key=k) → AntiJoin(Join(L, M, k), R, k)``
    when ``L``, ``M``, ``R`` are three distinct relations.

    This is the general case of the join-of-difference pattern. The
    existing :func:`_pass_join_difference_to_antijoin` handles the
    specialisation ``M = L`` (modulo projections), where the
    intermediate ``Join(L, M)`` collapses to ``L``. Here ``M`` is a
    different relation (e.g. ``Training_Enrollment`` while ``L`` is
    ``Employees``) and we keep both joins.

    Soundness:

    * ``Join(L, M − R)`` on ``k`` produces
      ``{l ∈ L | l.k ∈ ((M − R) projected onto k)}``,
      which under set semantics equals
      ``{l ∈ L | l.k ∈ M.k AND l.k ∉ R.k}``.
    * ``Join(L, M, k)`` produces L rows whose key is in M.k (possibly
      with duplicates if multiple M rows match — but the outer SELECT
      DISTINCT recovers set semantics).
    * ``AntiJoin(Join(L, M), R, k)`` then filters out the ones whose
      key is in R.k.
    * Combined: ``{l ∈ L | l.k ∈ M.k AND l.k ∉ R.k}`` — same set.

    Cardinality note: ``L ⋈ (M − R)`` on ``k`` (set EXCEPT) produces
    each surviving L row at most once, regardless of how many M rows
    match. The decomposed form ``Join(L, M)`` produces one L row per
    matching M row. To preserve set semantics, the outer SELECT
    needs ``DISTINCT``. The SQL converter already emits ``DISTINCT``
    when its ``distinct`` flag is set, and the planner's caller is
    responsible for setting that flag based on user intent. (The
    rule itself doesn't force ``DISTINCT`` — that's a downstream
    decision.)

    Conservatism: the rule does NOT fire when ``M = L`` (the existing
    pass handles that better, producing a cleaner output without the
    redundant ``Join(L, L)``). We detect that by stripping any
    projection chain on ``M`` and comparing to ``L``'s base; equal
    bases means we let the other pass handle it.
    """
    if not isinstance(node, OperatorNode):
        return node
    if not isinstance(node.params, JoinParams):
        return node
    if len(node.inputs) != 2:
        return node
    if not node.params.join_columns:
        return node

    left, right = node.inputs[0], node.inputs[1]
    if not isinstance(right, OperatorNode):
        return node
    if not isinstance(right.params, DifferenceParams):
        return node
    if len(right.inputs) != 2:
        return node

    diff_left, diff_right = right.inputs[0], right.inputs[1]

    # If M = L (modulo projection chains), the existing pass handles
    # it more cleanly. Bail so we don't double-rewrite.
    diff_left_base = _strip_projection_chain(diff_left)
    left_base = _strip_projection_chain(left)
    if _same_physical_relation(left_base, diff_left_base):
        return node

    # The three-way rewrite. We need to construct a regular Join(L, M, k)
    # with the column shape of Join(L, M). M's column shape comes from
    # diff_left's output_columns. The new output column shape is L's
    # columns extended with M's non-key columns (mirroring how the
    # operator-level Join builds its output).
    m_columns = (
        list(diff_left.output_columns)
        if isinstance(diff_left, OperatorNode) and diff_left.output_columns
        else (
            list(diff_left.columns) if isinstance(diff_left, TableLeafNode) else []
        )
    )
    l_columns = (
        list(left.output_columns)
        if isinstance(left, OperatorNode) and left.output_columns
        else (
            list(left.columns) if isinstance(left, TableLeafNode) else []
        )
    )
    if not m_columns or not l_columns:
        return node

    # Build the inner Join(L, M, k): output is L's columns followed by
    # M's non-join columns (mirrors the planner's join output
    # convention). Strip any projection chain on M so the inner Join
    # references the underlying table directly — the projection chain
    # only narrowed M's column shape, which the new Join's column list
    # supersedes.
    m_base = _strip_projection_chain(diff_left)
    join_keys = list(node.params.join_columns)
    m_extra = [c for c in m_columns if c not in join_keys]
    join_lm_output = list(l_columns) + m_extra

    join_lm = OperatorNode(
        operator="join",
        params=JoinParams(join_columns=join_keys),
        inputs=[left, m_base],
        output_columns=join_lm_output,
    )

    # AntiJoin's output column shape: it's a filter on its left, so
    # equal to the inner Join's output. The outer caller's
    # ``output_columns`` (which expects L's shape only) gets honoured
    # by the SQL converter's finalisation projection.
    return OperatorNode(
        operator="anti_join",
        params=AntiJoinParams(join_columns=join_keys),
        inputs=[join_lm, _strip_projection_chain(diff_right)],
        output_columns=join_lm_output,
        output_expression=node.output_expression,
    )


# ---------------------------------------------------------------------------
# Pass 5: drop a redundant filtering join inside a Join's right input
# ---------------------------------------------------------------------------


def _pass_drop_redundant_filtering_join(node: OperationNode) -> OperationNode:
    """``Join(L, π_cols(Join(A, B, key=k)), key=k) → Join(L, A, key=k)``
    when ``L.k ⊆ B.k`` is provable from L's tree provenance.

    Motivation. The planner often decorates a per-row relation ``L``
    that's already backed by a base table ``B`` (e.g. ``L`` is the
    ≥3-reviews self-join of ``Performance_Reviews``) by joining
    against ``π_{A.cols}(A ⋈ B)`` to attach ``A``'s columns
    (``first_name``/``last_name`` from ``Employees``). The role of
    the inner ``A ⋈ B`` is just to filter ``A`` to rows whose key has
    a match in ``B``. But ``L``'s key column already values entirely
    in ``B.k`` by construction, so the outer ``Join(L, …, k)`` would
    re-filter to ``L.k ∩ A.k ∩ B.k = L.k ∩ A.k`` regardless. The
    inner ``B``-filter is redundant.

    Soundness:

    * Original output rows are pairs ``(l, x)`` where ``l ∈ L``,
      ``x = (a, b) ∈ A ⋈ B`` (projected to ``A.cols``), ``l.k = x.k``.
      The output column shape is ``L.cols ∪ A.cols``.
    * Equivalent: ``(l, a) | l ∈ L, a ∈ A, l.k = a.k, l.k ∈ B.k``.
    * Since ``L.k ⊆ B.k``, the constraint ``l.k ∈ B.k`` is automatic
      for any ``l ∈ L`` — drop it. Result: ``(l, a) | l ∈ L, a ∈ A,
      l.k = a.k`` = ``Join(L, A, k)``.

    Provenance check. ``L.k ⊆ B.k`` is established by walking ``L``'s
    tree and verifying that every base-table contribution to the
    column ``k`` comes from ``B`` (by table name). See
    :func:`_key_provenance` for the exact recursion. The check is
    sound but conservative: it returns a *superset* of the actual
    table provenance for ``k``, so passes that fail the check might
    still be sound but won't fire.

    Out of scope: the rewrite preserves the ``π_cols`` projection
    that may sit between ``Join`` and ``A ⋈ B``, by re-attaching it
    as ``π_cols(A)`` over ``A`` directly. This keeps the column
    shape of the rewritten join's right input identical to the
    original.
    """
    if not isinstance(node, OperatorNode):
        return node
    if not isinstance(node.params, JoinParams):
        return node
    if len(node.inputs) != 2:
        return node
    if not node.params.join_columns:
        return node

    left, right = node.inputs[0], node.inputs[1]

    # The right input is typically wrapped in a projection
    # (``π_{A.cols}(A ⋈ B)``); peel it off so we can inspect the
    # underlying join. Track the projection so we can re-apply it to
    # the rewritten right input.
    right_projection: OperatorNode | None = None
    inner = right
    if (
        isinstance(inner, OperatorNode)
        and isinstance(inner.params, ProjectionParams)
        and inner.inputs
    ):
        right_projection = inner
        inner = inner.inputs[0]

    # Inner must be a binary Join.
    if not isinstance(inner, OperatorNode):
        return node
    if not isinstance(inner.params, JoinParams):
        return node
    if len(inner.inputs) != 2:
        return node
    if inner.params.join_columns != node.params.join_columns:
        # Inner join must be on the same key as the outer join.
        # Different keys mean the redundancy argument doesn't apply
        # (we'd be filtering A on a different relationship).
        return node

    a_node, b_node = inner.inputs[0], inner.inputs[1]

    # Provenance check on L: every base-table source for the join key
    # must be ``b_node``'s underlying table. Multiple keys: each must
    # individually pass the check.
    b_base_names = _base_table_names_for_columns(b_node, node.params.join_columns)
    if not b_base_names:
        return node
    l_provenance: dict[str, set[str]] = {}
    for k in node.params.join_columns:
        prov = _key_provenance(left, k)
        if prov is None:
            return node
        l_provenance[k] = prov

    # Every key column's L-provenance must be a subset of B's
    # base-table names.
    for k in node.params.join_columns:
        if not l_provenance[k] or not l_provenance[k].issubset(b_base_names[k]):
            return node

    # Rewrite: replace the outer join's right input with ``A`` (or
    # ``π_cols(A)`` when there was a projection). The new right input
    # exposes A's columns directly — no inner B-filter needed.
    if right_projection is not None:
        new_right: OperationNode = OperatorNode(
            operator="projection",
            params=ProjectionParams(columns=list(right_projection.params.columns)),
            inputs=[a_node],
            output_columns=list(right_projection.output_columns),
        )
    else:
        new_right = a_node

    return replace(node, inputs=[left, new_right])


def _base_table_names_for_columns(
    node: OperationNode, columns: list[str],
) -> dict[str, set[str]] | None:
    """For each column in ``columns``, return the set of base table
    names whose union covers ``node``'s values for that column.

    Returns ``None`` when any column can't be traced (e.g. comes from
    an aggregate or expression we can't reason about). The result is
    a *superset* of the actual provenance — sound for the
    "is-this-key-already-filtered-by-B" check.
    """
    out: dict[str, set[str]] = {}
    for k in columns:
        prov = _key_provenance(node, k)
        if prov is None:
            return None
        out[k] = prov
    return out


def _key_provenance(node: OperationNode, key: str) -> set[str] | None:
    """Return a *superset* bound on which base tables contribute
    values to ``node``'s column ``key``. ``None`` if the column can't
    be traced.

    For a sound "is-this-key-already-filtered-by-T" check, the caller
    verifies the returned set is a subset of ``{T.name}`` — meaning
    every value at ``node.key`` is provably in some base table the
    caller considers acceptable.

    Recursion:

    * ``TableLeaf(T)`` with ``T.cols`` containing ``key`` → ``{T.name}``;
      otherwise ``None``.
    * ``Selection(N)`` → ``key_provenance(N, key)`` (selection is a
      filter, doesn't introduce new values).
    * ``Projection(N)`` → ``key_provenance(N, key)`` if ``key`` is
      among the projection's columns, else ``None`` (the column was
      dropped). Aggregates / expressions — return ``None`` (we can't
      reason about derived columns at this level).
    * ``Join(N1, N2, k)`` natural-join semantics: output's ``key``
      values come from one or both sides. We pick the *smaller* of
      the two provenances (since output ``k`` ⊆ both), but only
      when ``key`` is the join key or appears on a single side.
    * ``CartesianProduct(N1, N2)`` — no key matching, output ``key``
      values come from whichever side the column belongs to.
    * ``Difference(N1, N2)`` → ``key_provenance(N1, key)`` (output
      values ⊆ N1).
    * ``Union(N1, N2)`` → union of both sides' provenances.
    * ``AntiJoin(N1, N2, k)`` → ``key_provenance(N1, key)`` (filter
      on N1).
    """
    if isinstance(node, TableLeafNode):
        if key in node.columns:
            return {node.table_name}
        return None

    if not isinstance(node, OperatorNode):
        return None

    params = node.params

    if isinstance(params, SelectionParams):
        if not node.inputs:
            return None
        return _key_provenance(node.inputs[0], key)

    if isinstance(params, ProjectionParams):
        if not node.inputs:
            return None
        # Only follow if ``key`` is preserved by the projection (i.e.
        # appears as a bare column among the projection's columns).
        # Aggregates and expressions block the trace.
        if key not in params.columns:
            return None
        return _key_provenance(node.inputs[0], key)

    if isinstance(params, JoinParams):
        if len(node.inputs) != 2:
            return None
        n1, n2 = node.inputs[0], node.inputs[1]
        # Output ``key`` values: when ``key`` is the join key, output
        # ``key`` ⊆ N1.k ∩ N2.k → use the tighter bound. When ``key``
        # is non-join, it sits on whichever side it came from
        # (the join doesn't intersect non-key columns).
        if key in params.join_columns:
            p1 = _key_provenance(n1, key)
            p2 = _key_provenance(n2, key)
            if p1 is None and p2 is None:
                return None
            if p1 is None:
                return p2
            if p2 is None:
                return p1
            # Take the smaller (tighter) bound.
            return p1 if len(p1) <= len(p2) else p2
        # Non-join column: try each side. Return the one that has it.
        p1 = _key_provenance(n1, key)
        if p1 is not None:
            return p1
        return _key_provenance(n2, key)

    if isinstance(params, CartesianProductParams):
        if len(node.inputs) != 2:
            return None
        # No join — output ``key`` values come from whichever side
        # actually has the column.
        p1 = _key_provenance(node.inputs[0], key)
        if p1 is not None:
            return p1
        return _key_provenance(node.inputs[1], key)

    if isinstance(params, DifferenceParams):
        if not node.inputs:
            return None
        return _key_provenance(node.inputs[0], key)

    if isinstance(params, UnionParams):
        if len(node.inputs) != 2:
            return None
        p1 = _key_provenance(node.inputs[0], key)
        p2 = _key_provenance(node.inputs[1], key)
        if p1 is None or p2 is None:
            return None
        return p1 | p2

    if isinstance(params, AntiJoinParams):
        if not node.inputs:
            return None
        return _key_provenance(node.inputs[0], key)

    return None


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
    _pass_join_difference_to_antijoin,
    _pass_difference_join_to_antijoin,
    _pass_join_three_way_difference_to_antijoin,
    _pass_drop_redundant_filtering_join,
]
