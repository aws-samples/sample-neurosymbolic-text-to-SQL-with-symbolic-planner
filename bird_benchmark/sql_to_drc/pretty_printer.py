"""DRC → SQL pretty-printer shim (Req 4.1).

Round-trip helper for the SQL → DRC → SQL → DRC equivalence property
(Property 13 in the design). The shim is intentionally thin: it
reuses the planner's existing :func:`text_to_sql_planner.sql.convert_query_to_sql`
rather than introducing a second SQL emitter inside ``bird_benchmark``.
The planner already knows how to render a DRC into SQL given the
correct OperationTree shape, so this module's only job is to construct
that OperationTree from a DRCExpression in the form that
:mod:`bird_benchmark.sql_to_drc.translator` produces.

Strategy
--------

Translator output for an in-scope SELECT has the following shape::

    QueryExpression =
        LIMIT? · ORDER_BY? · DRCExpression(
            result_variables = [...],
            condition = AND(
                MembershipNode(...),     # one per FROM/JOIN table
                MembershipNode(...),
                <WHERE / ON predicates>,
            ),
        )

The planner's happy path emits exactly the same shape via the SQL
converter when the OperationTree is::

    Selection(
        condition = <WHERE / ON predicates>,
        input    = Cartesian(... Cartesian(Leaf(t1), Leaf(t2)) ... Leaf(tn)),
    )

(or just the chain of cartesians when there is no remaining WHERE).
This module walks the top-level AND tree of the DRC condition,
peels every :class:`MembershipNode` into a :class:`TableLeafNode`
whose ``columns`` are the membership's bound variables (so the SQL
converter's variable resolver can find them in its ``var_map``), and
combines the leaves with cartesian-product nodes. The remaining
predicates are wrapped in a single ``Selection``.

Quantifiers (``EXISTS`` / ``IN (SELECT ...)``) live inside their own
sub-conditions; we deliberately do **not** descend into their bodies
here. The SQL converter has its own ``_quantifier_to_sql`` path that
handles the subquery shape inline.

Failure modes
-------------

When the DRC has no membership nodes at all (degenerate DRC, e.g.
``{x | true}`` produced by an empty parse), or the planner's SQL
converter rejects the constructed tree, we surface a structured
:class:`bird_benchmark.types.ConverterError` rather than raising. The
round-trip property test (task 15.2) will exercise both the success
and the structured-failure paths, mirroring the framework's "never
raise for malformed input" contract (Req 2.3 / Req 3).
"""

from __future__ import annotations

from typing import Union

from text_to_sql_planner.sql import (
    SQLFailure,
    SQLSuccess,
    convert_query_to_sql,
)
from text_to_sql_planner.types.drc import (
    DRCExpression,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    QuantifierNode,
    QueryExpression,
    query_inner_drc,
)
from text_to_sql_planner.types.operation_tree import (
    OperationNode,
    OperationTree,
    OperatorNode,
    TableLeafNode,
)
from text_to_sql_planner.types.operators import (
    CartesianProductParams,
    SelectionParams,
)

from ..types import ConverterError


__all__ = ["drc_to_sql"]


# --- Helpers ---------------------------------------------------------------


def _split_top_level_and(
    condition,
) -> tuple[list[MembershipNode], list]:
    """Walk top-level ``AND`` nodes; partition into memberships and rest.

    Stops at any non-AND node — in particular, it does **not** descend
    into ``NotNode``, ``OrNode``, comparisons, or quantifier bodies.
    A ``LiteralNode(value=1, data_type="number")`` (the trivially-true
    sentinel produced by :func:`bird_benchmark.sql_to_drc.translator._conjoin`
    for an empty FROM with no WHERE) is filtered out so it never reaches
    the SQL converter as a vacuous ``WHERE 1`` clause.
    """

    memberships: list[MembershipNode] = []
    others: list = []

    def visit(node) -> None:
        if isinstance(node, LogicalConnectiveNode) and node.operator == "and":
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, QuantifierNode) and node.kind == "exists":
            # The translator wraps the body in an outer existential
            # binding every FROM-clause variable that isn't a result
            # variable (so the formula has no free filter variables).
            # The pretty-printer treats those bindings transparently:
            # the bound names are exactly the ones that already appear
            # inside the body's memberships, so re-rendering them as
            # SELECT-list aliases works the same way regardless of
            # whether they're free or existentially bound. Descend
            # into the body and continue collecting.
            visit(node.body)
            return
        if isinstance(node, MembershipNode):
            memberships.append(node)
            return
        if (
            isinstance(node, LiteralNode)
            and node.data_type == "number"
            and node.value == 1
        ):
            # Trivially-true sentinel from ``_conjoin([])`` — drop it.
            return
        others.append(node)

    if condition is not None:
        visit(condition)
    return memberships, others


def _conjoin_others(parts: list):
    """Combine the remaining (non-membership) conjuncts back into a tree.

    Mirrors the translator's ``_conjoin`` helper but does not need to
    handle the empty case — callers of this helper only invoke it when
    ``parts`` is non-empty. Single-element input is returned as-is so
    we don't introduce a gratuitous one-arm ``and``.
    """

    result = parts[0]
    for nxt in parts[1:]:
        result = LogicalConnectiveNode(operator="and", left=result, right=nxt)
    return result


def _build_cartesian_chain(leaves: list[TableLeafNode]) -> OperationNode:
    """Combine ``leaves`` into a left-leaning cartesian-product chain.

    A single leaf is returned unchanged. Two or more leaves produce a
    chain ``CART(... CART(L1, L2) ..., Ln)`` whose ``output_columns``
    accumulate every leaf's columns in source order — this matches the
    planner's happy-path shape and lets the SQL converter resolve
    every membership variable through the chain's combined
    ``var_mapping``.
    """

    if len(leaves) == 1:
        return leaves[0]

    base: OperationNode = leaves[0]
    accumulated: list[str] = list(leaves[0].columns)
    for leaf in leaves[1:]:
        accumulated = accumulated + list(leaf.columns)
        base = OperatorNode(
            operator="cartesian_product",
            params=CartesianProductParams(),
            inputs=[base, leaf],
            output_columns=list(accumulated),
        )
    return base


# --- Public entrypoint -----------------------------------------------------


def drc_to_sql(expr: Union[QueryExpression, DRCExpression]) -> Union[str, ConverterError]:
    """Pretty-print a DRC query expression back to SQL.

    Parameters
    ----------
    expr:
        A :class:`~text_to_sql_planner.types.drc.QueryExpression` (the
        kind :func:`bird_benchmark.sql_to_drc.convert_sql` returns) or a
        bare :class:`~text_to_sql_planner.types.drc.DRCExpression`. Order
        BY / LIMIT wrappers are passed through to
        :func:`text_to_sql_planner.sql.convert_query_to_sql` unchanged so
        the outer SELECT picks them up.

    Returns
    -------
    str
        A SQL SELECT statement on success.
    ConverterError
        A structured error when the DRC has no membership nodes (no
        relations to FROM-clause), when the planner's SQL converter
        rejects the constructed OperationTree, or when an unexpected
        exception escapes the converter. The shim never raises.

    Notes
    -----
    The constructed OperationTree uses the membership variables as the
    leaf's ``columns`` (e.g. ``v_employees_id_1``) rather than the real
    table columns. This is deliberate — the DRC condition references
    those variable names via ``VariableRefNode``, and the SQL
    converter's variable resolver needs them present in its
    ``var_mapping`` to render qualified ``alias.<var_name>``
    references. The resulting SQL is well-formed but uses the
    synthetic variable names as if they were column names; the
    re-parse step in the round-trip property treats them as unknown
    identifiers and falls back to
    :func:`bird_benchmark.sql_to_drc.translator._collect_referenced_columns`,
    which is exactly the behaviour Property 13 (round-trip
    equivalence) was designed around.
    """

    try:
        # Strip ORDER BY / LIMIT wrappers to find the inner DRC; the
        # wrappers themselves are forwarded to ``convert_query_to_sql``
        # via the ``query`` argument so the outer SELECT picks them up.
        if isinstance(expr, DRCExpression):
            inner_drc: DRCExpression = expr
        else:
            inner_drc = query_inner_drc(expr)

        memberships, other_conjuncts = _split_top_level_and(inner_drc.condition)

        if not memberships:
            return ConverterError(
                kind="parse_error",
                message=(
                    "cannot pretty-print DRC without any top-level "
                    "MembershipNode: nothing to put in the FROM clause"
                ),
                line=1,
                column=1,
            )

        # One TableLeafNode per relation. The leaf's ``columns`` are the
        # membership variables themselves so the SQL converter's
        # variable resolver finds them when rendering the WHERE / SELECT.
        leaves = [
            TableLeafNode(
                table_name=m.relation,
                columns=list(m.variables),
            )
            for m in memberships
        ]

        base = _build_cartesian_chain(leaves)

        if other_conjuncts:
            condition = _conjoin_others(other_conjuncts)
            # ``output_columns`` for the selection mirrors the input's
            # — selection is a row filter, not a column rewrite.
            if isinstance(base, TableLeafNode):
                output_columns = list(base.columns)
            else:
                output_columns = list(base.output_columns)
            root: OperationNode = OperatorNode(
                operator="selection",
                params=SelectionParams(condition=condition),
                inputs=[base],
                output_columns=output_columns,
            )
        else:
            root = base

        tree = OperationTree(root=root)
        result = convert_query_to_sql(tree, expr, distinct=False)

        if isinstance(result, SQLSuccess):
            return result.sql

        # ``SQLFailure`` from the planner's converter — surface its
        # message through the structured error channel so the round-
        # trip harness can record it as one of the four artifacts.
        assert isinstance(result, SQLFailure)
        return ConverterError(
            kind="parse_error",
            message=f"DRC→SQL converter failed: {result.error}",
            line=1,
            column=1,
        )

    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        # Mirrors the contract in :func:`bird_benchmark.sql_to_drc.convert_sql`:
        # never raise for malformed input. Any exception escaping the
        # planner's SQL converter is converted into a structured error
        # so the suite keeps running.
        return ConverterError(
            kind="parse_error",
            message=(
                f"pretty-printer raised {type(exc).__name__}: {exc}"
            ),
            line=1,
            column=1,
        )
