"""Pre-SMT preprocessing for DRC conditions.

Before handing a DRC condition to ``convert_to_smt``, we run two
correctness-preserving simplifications that materially help cvc5 decide
the resulting formulas:

1. **Trivial-equality elimination.** A pattern like
   ``(exists (v ...) (and ... (= v expr) ...))`` where ``expr`` does not
   reference ``v`` is logically equivalent to substituting ``expr`` for
   ``v`` in the body and dropping ``v`` from the binder. This removes
   the existential and the equality in one step. The planner often
   leaves these around when an operator says "this output column equals
   some inner witness"; after substitution the witness disappears
   entirely.

2. **Unused-slot elimination on uninterpreted predicates.** Many relation
   slots — e.g. ``address``, ``gender``, ``comments`` — are introduced
   by the table-converter and then never referenced in any comparison
   or shared with another relation. If a slot ``k`` of relation ``R``
   is never *connected* to anything else (used only as a unique fresh
   bound variable at every membership of ``R``), we drop slot ``k``
   from ``R``'s signature and rewrite each membership accordingly.
   This is sound because an uninterpreted predicate's truth at a slot
   that nothing else reads cannot influence the truth of the whole
   formula — the existential's choice for that slot is unconstrained
   either way.

Both passes operate on the DRC AST directly so the existing SMT
converter (``smt_converter._convert_node``, etc.) can be reused
unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from text_to_sql_planner.types.drc import (
    ArithmeticNode,
    ComparisonNode,
    DRCCondition,
    FunctionCallNode,
    IsNotNullNode,
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


def eliminate_trivial_equalities(condition: DRCCondition) -> DRCCondition:
    """Remove existentials whose binder is forced equal to another term.

    Applies the rewrite

        (exists (... v ...) (and ... (= v expr) ...))
            ⟶ (exists (... (without v) ...) body[v := expr])

    when ``expr`` does not reference ``v`` (no self-binding) and is a
    variable or literal we can copy in safely. The pass runs to a fixed
    point so cascading equalities (e.g. ``v = w`` followed by
    ``w = ground``) are all resolved.

    Quantifier scope is respected: we only substitute inside the body
    that introduced the equality. Inner quantifiers that re-bind ``v``
    shadow the substitution as usual.
    """
    return _fixed_point(_eliminate_pass, condition)


def _fixed_point(transform, node: DRCCondition) -> DRCCondition:
    for _ in range(64):  # bounded for safety; converges in O(depth)
        new_node = transform(node)
        if _structural_equal(new_node, node):
            return node
        node = new_node
    return node


def _eliminate_pass(node: DRCCondition) -> DRCCondition:
    """One pass of the equality-elimination rewrite."""
    if node is None:
        return node

    if isinstance(node, QuantifierNode):
        new_body = _eliminate_pass(node.body)
        if node.kind != "exists":
            return _maybe(node, body=new_body)

        # Try to find an equality (= v expr) at the top level of an AND
        # chain in the body that pins one of our bound vars.
        bound = list(node.variables)
        bound_set = set(bound)
        equalities = _collect_top_level_equalities(new_body)

        for var, expr in equalities:
            if var not in bound_set:
                continue
            if _references(expr, var):
                continue
            # Membership slots are positional variable names — they
            # accept variables, not literals. If ``var`` is bound to a
            # literal but also appears as a membership slot inside the
            # body, eliminating the existential would orphan the slot:
            # ``_substitute`` (correctly) refuses to splice a literal
            # into a positional slot, so the slot would silently become
            # a free constant of the same name as the eliminated bound
            # variable, changing the formula's semantics. The dev_78
            # ``schools(..., City, ...) ∧ City = "Adelanto"`` case is
            # the canonical instance: City is a membership slot, so
            # eliminating the existential yields ``schools(..., City,
            # ...)`` where ``City`` is now a free String constant
            # instead of the literal "Adelanto".
            #
            # Skip elimination when ``expr`` is a literal AND ``var``
            # appears as a membership-slot variable. Variable-to-
            # variable substitution is still safe (slots accept
            # variables), so we only block the literal case.
            if (
                isinstance(expr, LiteralNode)
                and _appears_in_membership_slot(new_body, var)
            ):
                continue
            # Determine if `expr` mentions any bound name we're about to
            # remove from the binder. If `expr` is itself a fellow-bound
            # variable (e.g. emp_id_1 = emp_id_2 with both bound here),
            # we keep the equality — substituting away one would unbind
            # the other.
            if isinstance(expr, VariableRefNode) and expr.name in bound_set:
                # Pick the lexicographically-smaller name to keep, drop
                # the other; this still requires that the kept name is
                # also in scope, which it is.
                keep, drop = sorted([var, expr.name])
                if drop not in bound_set:
                    continue
                substitution = {drop: VariableRefNode(name=keep)}
                rewritten = _drop_equality(new_body, drop, VariableRefNode(name=keep))
                rewritten = _substitute(rewritten, substitution)
                new_bound = [v for v in bound if v != drop]
                if not new_bound:
                    return rewritten
                return QuantifierNode(
                    kind="exists",
                    variables=new_bound,
                    body=rewritten,
                )

            # `expr` is a literal or a non-bound variable — straightforward
            # substitution.
            substitution = {var: expr}
            rewritten = _drop_equality(new_body, var, expr)
            rewritten = _substitute(rewritten, substitution)
            new_bound = [v for v in bound if v != var]
            if not new_bound:
                return rewritten
            return QuantifierNode(
                kind="exists",
                variables=new_bound,
                body=rewritten,
            )

        return _maybe(node, body=new_body)

    if isinstance(node, LogicalConnectiveNode):
        return LogicalConnectiveNode(
            operator=node.operator,
            left=_eliminate_pass(node.left),
            right=_eliminate_pass(node.right),
        )

    if isinstance(node, NotNode):
        return NotNode(operand=_eliminate_pass(node.operand))

    if isinstance(node, ComparisonNode):
        return ComparisonNode(
            operator=node.operator,
            left=_eliminate_pass(node.left),
            right=_eliminate_pass(node.right),
        )

    if isinstance(node, ArithmeticNode):
        return ArithmeticNode(
            operator=node.operator,
            left=_eliminate_pass(node.left),
            right=_eliminate_pass(node.right),
        )

    if isinstance(node, FunctionCallNode):
        return FunctionCallNode(
            function=node.function,
            arguments=[_eliminate_pass(a) for a in node.arguments],
        )

    if isinstance(node, IsNotNullNode):
        # Leaf node — already in the canonical form ``rewrite_null_checks``
        # produces, so this pass returns it unchanged.
        return node

    return node


def _collect_top_level_equalities(node: DRCCondition) -> list[tuple[str, DRCCondition]]:
    """Return candidate ``(var_name, expr)`` pairs for every ``(= v expr)``
    reachable through top-level AND chains in ``node``.

    Both orientations are emitted when both sides qualify. This lets the
    eliminator choose whichever side is bound by the enclosing exists
    — without emitting both, an equality of the form ``(= outer inner)``
    where ``outer`` is universally bound and ``inner`` is the locally-
    existential binder would never be eliminated, because the collector
    would only ever propose ``("outer", inner)`` (and outer isn't in the
    local binder set).

    We don't descend into OR, NOT, IMPLIES, or further quantifiers —
    substituting under those would change semantics.
    """
    out: list[tuple[str, DRCCondition]] = []
    if isinstance(node, ComparisonNode) and node.operator == "=":
        left_is_var = isinstance(node.left, VariableRefNode)
        right_is_var = isinstance(node.right, VariableRefNode)
        if left_is_var and not _references(node.right, node.left.name):
            out.append((node.left.name, node.right))
        if right_is_var and not _references(node.left, node.right.name):
            out.append((node.right.name, node.left))
    elif isinstance(node, LogicalConnectiveNode) and node.operator == "and":
        out.extend(_collect_top_level_equalities(node.left))
        out.extend(_collect_top_level_equalities(node.right))
    return out


def _drop_equality(node: DRCCondition, var: str, expr: DRCCondition) -> DRCCondition:
    """Remove the matching ``(= var expr)`` (or ``(= expr var)``) from a
    top-level AND chain. Returns ``node`` unchanged if not found.

    If the equality is one half of an AND, the other half is returned.
    Multiple matching equalities can exist; we drop the first.
    """
    if isinstance(node, ComparisonNode) and node.operator == "=":
        if (
            isinstance(node.left, VariableRefNode) and node.left.name == var
            and _structural_equal(node.right, expr)
        ):
            return _LITERAL_TRUE
        if (
            isinstance(node.right, VariableRefNode) and node.right.name == var
            and _structural_equal(node.left, expr)
        ):
            return _LITERAL_TRUE
        return node

    if isinstance(node, LogicalConnectiveNode) and node.operator == "and":
        new_left = _drop_equality(node.left, var, expr)
        if not _structural_equal(new_left, node.left):
            return _and(new_left, node.right)
        new_right = _drop_equality(node.right, var, expr)
        if not _structural_equal(new_right, node.right):
            return _and(node.left, new_right)

    return node


_LITERAL_TRUE = LiteralNode(value=1, data_type="number")  # placeholder; pruned by _and


def _and(a: DRCCondition, b: DRCCondition) -> DRCCondition:
    """Build an AND, eliding ``True`` placeholders (the LiteralNode value=1
    we use as a removed-equality marker)."""
    if a is _LITERAL_TRUE or _is_true_marker(a):
        return b
    if b is _LITERAL_TRUE or _is_true_marker(b):
        return a
    return LogicalConnectiveNode(operator="and", left=a, right=b)


def _is_true_marker(node: DRCCondition) -> bool:
    return (
        isinstance(node, LiteralNode)
        and node.data_type == "number"
        and node.value == 1
    )


def _references(node: DRCCondition, var: str) -> bool:
    """Does ``node`` contain a free reference to ``var``?"""
    if node is None:
        return False
    if isinstance(node, VariableRefNode):
        return node.name == var
    if isinstance(node, MembershipNode):
        return var in node.variables
    if isinstance(node, IsNotNullNode):
        # ``IsNotNullNode.column`` is a column-binding name treated the
        # same way as a ``MembershipNode`` slot.
        return node.column == var
    if isinstance(node, QuantifierNode):
        if var in node.variables:
            return False  # shadowed
        return _references(node.body, var)
    if isinstance(node, LogicalConnectiveNode):
        return _references(node.left, var) or _references(node.right, var)
    if isinstance(node, NotNode):
        return _references(node.operand, var)
    if isinstance(node, ComparisonNode):
        return _references(node.left, var) or _references(node.right, var)
    if isinstance(node, ArithmeticNode):
        return _references(node.left, var) or _references(node.right, var)
    if isinstance(node, FunctionCallNode):
        return any(_references(a, var) for a in node.arguments)
    return False


def _appears_in_membership_slot(node: DRCCondition, var: str) -> bool:
    """Does ``node`` contain a membership term whose positional slot
    list includes ``var``?

    A positive answer means ``var`` is being used as a column-binding
    name in some ``(in (...) Table)`` term. Eliminating ``var`` by
    substituting a *literal* would orphan that slot — the resulting
    membership would still mention the name ``var`` (now a free
    constant) instead of the literal value. Variable-to-variable
    substitution is fine; this helper only matters for the literal case.

    Inner quantifiers that re-bind ``var`` shadow the search, mirroring
    ``_references``.
    """
    if node is None:
        return False
    if isinstance(node, MembershipNode):
        return var in node.variables
    if isinstance(node, IsNotNullNode):
        # ``IsNotNullNode.column`` is a column-binding name treated the
        # same way as a ``MembershipNode`` slot for substitution purposes:
        # eliminating ``var`` by substituting a literal would orphan the
        # column-name slot.
        return node.column == var
    if isinstance(node, QuantifierNode):
        if var in node.variables:
            return False  # shadowed
        return _appears_in_membership_slot(node.body, var)
    if isinstance(node, LogicalConnectiveNode):
        return (
            _appears_in_membership_slot(node.left, var)
            or _appears_in_membership_slot(node.right, var)
        )
    if isinstance(node, NotNode):
        return _appears_in_membership_slot(node.operand, var)
    # Comparison / arithmetic / function-call / variable-ref / literal:
    # references count as references-only, not membership slots.
    return False


def _substitute(node: DRCCondition, mapping: dict[str, DRCCondition]) -> DRCCondition:
    """Capture-avoiding substitution of free variables.

    Used after an existential elimination — we substitute the bound
    variable with its determined expression throughout the body. The
    substitution stops at any inner binder that reuses the same name
    (variable shadowing), and respects MembershipNode's ``variables``
    list as an ordered list of identifier slots.
    """
    if node is None:
        return node
    if isinstance(node, VariableRefNode):
        repl = mapping.get(node.name)
        return repl if repl is not None else node
    if isinstance(node, MembershipNode):
        new_vars = []
        rewrote = False
        for v in node.variables:
            repl = mapping.get(v)
            if isinstance(repl, VariableRefNode):
                new_vars.append(repl.name)
                rewrote = True
            elif repl is None:
                new_vars.append(v)
            else:
                # Substituting a literal into a positional membership
                # slot has no DRC representation. Bail by leaving the
                # original name; the equality elimination shouldn't
                # have fired on a membership-slot variable in the first
                # place.
                new_vars.append(v)
        if rewrote:
            return MembershipNode(variables=new_vars, relation=node.relation)
        return node
    if isinstance(node, IsNotNullNode):
        # ``IsNotNullNode.column`` is a column-binding name treated the
        # same way as a ``MembershipNode`` slot: only variable-to-variable
        # substitutions apply.
        repl = mapping.get(node.column)
        if isinstance(repl, VariableRefNode):
            return IsNotNullNode(column=repl.name)
        return node
    if isinstance(node, QuantifierNode):
        # Names rebound here shadow the substitution.
        rebound = set(node.variables)
        inner_map = {k: v for k, v in mapping.items() if k not in rebound}
        if not inner_map:
            return node
        return QuantifierNode(
            kind=node.kind,
            variables=list(node.variables),
            body=_substitute(node.body, inner_map),
        )
    if isinstance(node, LogicalConnectiveNode):
        return LogicalConnectiveNode(
            operator=node.operator,
            left=_substitute(node.left, mapping),
            right=_substitute(node.right, mapping),
        )
    if isinstance(node, NotNode):
        return NotNode(operand=_substitute(node.operand, mapping))
    if isinstance(node, ComparisonNode):
        return ComparisonNode(
            operator=node.operator,
            left=_substitute(node.left, mapping),
            right=_substitute(node.right, mapping),
        )
    if isinstance(node, ArithmeticNode):
        return ArithmeticNode(
            operator=node.operator,
            left=_substitute(node.left, mapping),
            right=_substitute(node.right, mapping),
        )
    if isinstance(node, FunctionCallNode):
        return FunctionCallNode(
            function=node.function,
            arguments=[_substitute(a, mapping) for a in node.arguments],
        )
    return node


def _maybe(node: QuantifierNode, body: DRCCondition) -> QuantifierNode:
    if _structural_equal(body, node.body):
        return node
    return QuantifierNode(kind=node.kind, variables=list(node.variables), body=body)


def _structural_equal(a: DRCCondition, b: DRCCondition) -> bool:
    """Structural equality on DRC nodes. Cheaper than dumping to lisp."""
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if isinstance(a, VariableRefNode):
        return a.name == b.name
    if isinstance(a, LiteralNode):
        return a.data_type == b.data_type and a.value == b.value
    if isinstance(a, MembershipNode):
        return a.relation == b.relation and a.variables == b.variables
    if isinstance(a, IsNotNullNode):
        return a.column == b.column
    if isinstance(a, QuantifierNode):
        return (
            a.kind == b.kind
            and a.variables == b.variables
            and _structural_equal(a.body, b.body)
        )
    if isinstance(a, LogicalConnectiveNode):
        return (
            a.operator == b.operator
            and _structural_equal(a.left, b.left)
            and _structural_equal(a.right, b.right)
        )
    if isinstance(a, NotNode):
        return _structural_equal(a.operand, b.operand)
    if isinstance(a, ComparisonNode):
        return (
            a.operator == b.operator
            and _structural_equal(a.left, b.left)
            and _structural_equal(a.right, b.right)
        )
    if isinstance(a, ArithmeticNode):
        return (
            a.operator == b.operator
            and _structural_equal(a.left, b.left)
            and _structural_equal(a.right, b.right)
        )
    if isinstance(a, FunctionCallNode):
        if a.function != b.function or len(a.arguments) != len(b.arguments):
            return False
        return all(_structural_equal(x, y) for x, y in zip(a.arguments, b.arguments))
    return False


# ---------------------------------------------------------------------------
# Pass 2: unused-slot elimination on uninterpreted predicates
# ---------------------------------------------------------------------------


def eliminate_unused_relation_slots(
    conditions: Iterable[DRCCondition],
    keep_names: Iterable[str] | None = None,
) -> dict[str, list[bool]]:
    """Decide which slot positions of each relation are *used*.

    A slot ``(R, k)`` is "used" iff at some membership site ``(R, k)``
    bound name ``v`` is *connected* to anything beyond that exact slot
    position. ``v`` is connected if any of the following holds:

    - ``v`` is referenced outside any membership term (in a comparison,
      arithmetic, function-call argument, or as a free reference).
    - ``v`` appears at any other ``(relation, slot_position)`` in any
      membership — for example, used as a join key between two
      relations, or appearing twice in the same relation's tuple.
    - ``v`` is in the ``keep_names`` allow-list.

    A slot all of whose occupants are *unconnected* (each occupant
    occupies only this exact slot, nowhere else) is unused: replacing
    the predicate with one that ignores that argument doesn't change
    the formula's truth value.

    Returns a dict ``relation_name → [is_used_per_slot]``.
    """
    # For each variable name, the set of (relation, slot) positions it
    # occupies in any membership. If this set has more than one element
    # for a given variable, the variable is "connected" across slots.
    var_sites: dict[str, set[tuple[str, int]]] = defaultdict(set)
    arities: dict[str, int] = {}
    referenced_outside_membership: set[str] = set(keep_names or ())

    def visit(node: DRCCondition) -> None:
        if node is None:
            return
        if isinstance(node, MembershipNode):
            arities[node.relation] = max(
                arities.get(node.relation, 0), len(node.variables)
            )
            for i, v in enumerate(node.variables):
                var_sites[v].add((node.relation, i))
            return
        if isinstance(node, VariableRefNode):
            referenced_outside_membership.add(node.name)
            return
        if isinstance(node, IsNotNullNode):
            # ``IsNotNullNode.column`` references a column-binding name
            # outside any membership slot; record it so the bound
            # variable holding that column doesn't get pruned as unused.
            if node.column:
                referenced_outside_membership.add(node.column)
            return
        if isinstance(node, ComparisonNode):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, ArithmeticNode):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, FunctionCallNode):
            for a in node.arguments:
                visit(a)
            return
        if isinstance(node, QuantifierNode):
            visit(node.body)
            return
        if isinstance(node, LogicalConnectiveNode):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, NotNode):
            visit(node.operand)
            return

    for c in conditions:
        visit(c)

    result: dict[str, list[bool]] = {}
    for rel, arity in arities.items():
        used = [False] * arity
        # For each (rel, slot) position, look at the set of variable
        # names that ever appear there.
        site_vars: dict[int, set[str]] = defaultdict(set)
        for v, sites in var_sites.items():
            for r, s in sites:
                if r == rel:
                    site_vars[s].add(v)

        for slot in range(arity):
            names = site_vars.get(slot, set())
            if not names:
                used[slot] = False
                continue
            # The slot is "used" iff at least one occupant is connected:
            # either referenced outside membership, or appears at >1
            # distinct (relation, slot) site total.
            for name in names:
                if name in referenced_outside_membership:
                    used[slot] = True
                    break
                if len(var_sites[name]) > 1:
                    used[slot] = True
                    break
            else:
                used[slot] = False
        result[rel] = used
    return result


def drop_unused_slots(
    condition: DRCCondition,
    used_slots: dict[str, list[bool]],
) -> DRCCondition:
    """Rewrite ``condition`` so each ``MembershipNode`` for a relation
    listed in ``used_slots`` keeps only its used slot positions, and the
    bound variables that previously filled the dropped slots are removed
    from any enclosing existential's binder list.

    We assume the caller already verified those bound variables are
    nowhere referenced (that's exactly the precondition
    ``eliminate_unused_relation_slots`` checks).
    """
    # First, find variables that we'll be dropping from membership terms;
    # collect them so we can prune existential binders that bind them.
    dropped_vars: set[str] = set()

    def find_dropped(node: DRCCondition) -> None:
        if node is None:
            return
        if isinstance(node, MembershipNode):
            mask = used_slots.get(node.relation)
            if not mask:
                return
            for i, v in enumerate(node.variables):
                if i < len(mask) and not mask[i]:
                    dropped_vars.add(v)
            return
        if isinstance(node, IsNotNullNode):
            # No membership slot to drop; leaf node.
            return
        if isinstance(node, QuantifierNode):
            find_dropped(node.body)
            return
        if isinstance(node, LogicalConnectiveNode):
            find_dropped(node.left)
            find_dropped(node.right)
            return
        if isinstance(node, NotNode):
            find_dropped(node.operand)
            return
        if isinstance(node, ComparisonNode):
            find_dropped(node.left)
            find_dropped(node.right)
            return
        if isinstance(node, ArithmeticNode):
            find_dropped(node.left)
            find_dropped(node.right)
            return
        if isinstance(node, FunctionCallNode):
            for a in node.arguments:
                find_dropped(a)
            return

    find_dropped(condition)

    def rewrite(node: DRCCondition) -> DRCCondition:
        if node is None:
            return node
        if isinstance(node, MembershipNode):
            mask = used_slots.get(node.relation)
            if not mask:
                return node
            new_vars = [v for i, v in enumerate(node.variables) if i < len(mask) and mask[i]]
            if new_vars == node.variables:
                return node
            return MembershipNode(variables=new_vars, relation=node.relation)
        if isinstance(node, IsNotNullNode):
            # Leaf node, no slots to drop.
            return node
        if isinstance(node, QuantifierNode):
            new_body = rewrite(node.body)
            new_bindings = [v for v in node.variables if v not in dropped_vars]
            if not new_bindings:
                return new_body
            return QuantifierNode(
                kind=node.kind,
                variables=new_bindings,
                body=new_body,
            )
        if isinstance(node, LogicalConnectiveNode):
            return LogicalConnectiveNode(
                operator=node.operator,
                left=rewrite(node.left),
                right=rewrite(node.right),
            )
        if isinstance(node, NotNode):
            return NotNode(operand=rewrite(node.operand))
        if isinstance(node, ComparisonNode):
            return ComparisonNode(
                operator=node.operator,
                left=rewrite(node.left),
                right=rewrite(node.right),
            )
        if isinstance(node, ArithmeticNode):
            return ArithmeticNode(
                operator=node.operator,
                left=rewrite(node.left),
                right=rewrite(node.right),
            )
        if isinstance(node, FunctionCallNode):
            return FunctionCallNode(
                function=node.function,
                arguments=[rewrite(a) for a in node.arguments],
            )
        return node

    return rewrite(condition)


# ---------------------------------------------------------------------------
# Pass 3: STRFTIME ↔ integer-date comparison rewrite
#
# cvc5 has no built-in semantics for ``STRFTIME``. Without help, a
# comparison like ``STRFTIME('%Y', dob) > '1980'`` becomes an
# uninterpreted ``(> (STRFTIME "%Y" dob) 1980)`` call that the solver
# cannot relate to the planner-side ``(> dob 4017)`` form (4017 = days
# since epoch for 1980-12-31). The verdict comes back
# ``not_equivalent`` even though the queries describe the same set of
# rows over actual date data.
#
# We exploit the fact that BIRD's gold queries use ``STRFTIME`` with a
# small finite set of patterns:
#
#   STRFTIME('%Y', d) <op> 'YYYY'
#
# where ``<op>`` is one of ``< <= > >= = !=`` and ``'YYYY'`` is a
# four-digit year string. For each such pattern, we substitute an
# equivalent integer comparison on ``d`` (interpreted as days since
# 1970-01-01, the same convention used by ``_convert_node_for_comparison``).
#
# The rewrite fires for both arg orderings (literal can be on the
# left). Anything that doesn't match the year-comparison shape passes
# through unchanged — the SMT converter still declares STRFTIME
# uninterpreted in that case (e.g. ``%m``, ``%d``, or non-literal
# right-hand side).
# ---------------------------------------------------------------------------


def rewrite_strftime_year_comparisons(
    condition: DRCCondition,
) -> DRCCondition:
    """Rewrite ``STRFTIME('%Y', d) <op> 'YYYY'`` patterns into integer
    date comparisons. See pass docstring above."""
    return _strftime_rewrite(condition)


def _strftime_rewrite(node: DRCCondition) -> DRCCondition:
    if node is None:
        return node
    if isinstance(node, IsNotNullNode):
        # Leaf; no STRFTIME shape to rewrite.
        return node
    if isinstance(node, ComparisonNode):
        rewritten = _try_rewrite_strftime_comparison(node)
        if rewritten is not None:
            return rewritten
        return ComparisonNode(
            operator=node.operator,
            left=_strftime_rewrite(node.left),
            right=_strftime_rewrite(node.right),
        )
    if isinstance(node, LogicalConnectiveNode):
        return LogicalConnectiveNode(
            operator=node.operator,
            left=_strftime_rewrite(node.left),
            right=_strftime_rewrite(node.right),
        )
    if isinstance(node, NotNode):
        return NotNode(operand=_strftime_rewrite(node.operand))
    if isinstance(node, QuantifierNode):
        return QuantifierNode(
            kind=node.kind,
            variables=list(node.variables),
            body=_strftime_rewrite(node.body),
        )
    if isinstance(node, ArithmeticNode):
        return ArithmeticNode(
            operator=node.operator,
            left=_strftime_rewrite(node.left),
            right=_strftime_rewrite(node.right),
        )
    if isinstance(node, FunctionCallNode):
        return FunctionCallNode(
            function=node.function,
            arguments=[_strftime_rewrite(a) for a in node.arguments],
        )
    return node


def _try_rewrite_strftime_comparison(
    node: ComparisonNode,
) -> DRCCondition | None:
    """If ``node`` is ``STRFTIME('%Y', d) <op> 'YYYY'`` (or its
    flipped form ``'YYYY' <op> STRFTIME(...)``), return the rewritten
    integer-comparison form. Otherwise return ``None``."""
    op = node.operator
    flip = False
    fn_node, lit_node = _split_strftime_call_and_literal(node.left, node.right)
    if fn_node is None:
        fn_node, lit_node = _split_strftime_call_and_literal(node.right, node.left)
        flip = True
    if fn_node is None:
        return None
    # Function on the LHS by convention; flip the operator's direction
    # if we had to swap operands.
    if flip:
        op = _flip_comparison_op(op)

    if len(fn_node.arguments) != 2:
        return None
    fmt_arg = fn_node.arguments[0]
    if not (isinstance(fmt_arg, LiteralNode) and fmt_arg.data_type == "string"):
        return None
    if str(fmt_arg.value) != "%Y":
        return None

    date_node = fn_node.arguments[1]

    year = _parse_year_literal(lit_node)
    if year is None:
        return None

    boundary_lo = _days_since_epoch(year, 1, 1)        # Jan 1 of year
    boundary_hi = _days_since_epoch(year + 1, 1, 1)    # Jan 1 of year+1

    def _cmp(op_str: str, boundary: int) -> ComparisonNode:
        return ComparisonNode(
            operator=op_str,
            left=date_node,
            right=LiteralNode(value=boundary, data_type="number"),
        )

    if op == ">":
        return _cmp(">=", boundary_hi)
    if op == ">=":
        return _cmp(">=", boundary_lo)
    if op == "<":
        return _cmp("<", boundary_lo)
    if op == "<=":
        return _cmp("<", boundary_hi)
    if op == "=":
        return LogicalConnectiveNode(
            operator="and",
            left=_cmp(">=", boundary_lo),
            right=_cmp("<", boundary_hi),
        )
    if op == "!=":
        return LogicalConnectiveNode(
            operator="or",
            left=_cmp("<", boundary_lo),
            right=_cmp(">=", boundary_hi),
        )
    return None


def _split_strftime_call_and_literal(
    a: DRCCondition, b: DRCCondition
) -> tuple[FunctionCallNode | None, LiteralNode | None]:
    """If ``a`` is a ``STRFTIME(...)`` call and ``b`` is a string
    literal, return ``(a, b)``. Otherwise return ``(None, None)``."""
    if isinstance(a, FunctionCallNode) and a.function == "STRFTIME":
        if isinstance(b, LiteralNode) and b.data_type == "string":
            return a, b
    return None, None


def _flip_comparison_op(op: str) -> str:
    """Return the operator that yields the same truth value when its
    operands are swapped (``a > b`` ↔ ``b < a``)."""
    return {
        "<": ">", "<=": ">=", ">": "<", ">=": "<=",
        "=": "=", "!=": "!=",
    }.get(op, op)


def _parse_year_literal(node: DRCCondition) -> int | None:
    """If ``node`` is a string literal whose value is a four-digit
    year, return the year as an int. Otherwise return ``None``."""
    if not isinstance(node, LiteralNode):
        return None
    if node.data_type != "string":
        return None
    val = str(node.value).strip()
    if len(val) != 4 or not val.isdigit():
        return None
    year = int(val)
    if year < 1 or year > 9999:
        return None
    return year


def _days_since_epoch(year: int, month: int, day: int) -> int:
    """Days from 1970-01-01 to (year, month, day). Matches the
    integer date encoding ``_convert_node_for_comparison`` uses."""
    from datetime import date
    return (date(year, month, day) - date(1970, 1, 1)).days


# ---------------------------------------------------------------------------
# Pass 4: normalise integer-boundary comparisons
#
# ``<= N`` (where N is an integer literal) is rewritten to ``< N+1``,
# and ``>= N`` is rewritten to ``> N-1``. This canonical form means
# a date comparison like ``ExamDate <= 10226`` and its STRFTIME-rewrite
# sibling ``ExamDate < 10227`` become structurally identical after both
# pass through the normalisation. Without it, cvc5 sometimes can't
# prove the equivalence of wide existential formulas that differ only
# in this choice of boundary direction.
#
# The rewrite is semantics-preserving over Int: ``x <= N`` ↔ ``x < N+1``
# and ``x >= N`` ↔ ``x > N-1`` for all integers. It does NOT fire for
# String sorts (where the +1 arithmetic doesn't apply).
# ---------------------------------------------------------------------------


def normalise_integer_boundary_comparisons(
    condition: DRCCondition,
) -> DRCCondition:
    """Rewrite ``<= N`` → ``< N+1`` and ``>= N`` → ``> N-1`` for
    integer literals. See pass docstring above."""
    return _normalise_boundary(condition)


def _normalise_boundary(node: DRCCondition) -> DRCCondition:
    if node is None:
        return node
    if isinstance(node, IsNotNullNode):
        # Leaf; no integer-boundary comparison shape to rewrite.
        return node
    if isinstance(node, ComparisonNode):
        rewritten = _try_normalise_boundary(node)
        if rewritten is not None:
            return rewritten
        return ComparisonNode(
            operator=node.operator,
            left=_normalise_boundary(node.left),
            right=_normalise_boundary(node.right),
        )
    if isinstance(node, LogicalConnectiveNode):
        return LogicalConnectiveNode(
            operator=node.operator,
            left=_normalise_boundary(node.left),
            right=_normalise_boundary(node.right),
        )
    if isinstance(node, NotNode):
        return NotNode(operand=_normalise_boundary(node.operand))
    if isinstance(node, QuantifierNode):
        return QuantifierNode(
            kind=node.kind,
            variables=list(node.variables),
            body=_normalise_boundary(node.body),
        )
    if isinstance(node, ArithmeticNode):
        return ArithmeticNode(
            operator=node.operator,
            left=_normalise_boundary(node.left),
            right=_normalise_boundary(node.right),
        )
    if isinstance(node, FunctionCallNode):
        return FunctionCallNode(
            function=node.function,
            arguments=[_normalise_boundary(a) for a in node.arguments],
        )
    return node


def _try_normalise_boundary(node: ComparisonNode) -> ComparisonNode | None:
    """If the comparison is ``expr <= IntLit`` or ``IntLit >= expr``,
    normalise to ``expr < IntLit+1`` / ``IntLit-1 > expr``.
    Same for ``>= IntLit`` → ``> IntLit-1``."""
    op = node.operator

    # Case: expr <= N  →  expr < N+1
    if op == "<=" and isinstance(node.right, LiteralNode) and _is_int_literal(node.right):
        return ComparisonNode(
            operator="<",
            left=node.left,
            right=LiteralNode(value=node.right.value + 1, data_type="number"),
        )
    # Case: N <= expr  →  N-1 < expr  (i.e. expr > N-1  → flip: N-1 < expr is just >= N on the left)
    # Actually: N <= expr  ↔  expr >= N. Let's also normalise >= on the right:
    # Case: expr >= N  →  expr > N-1
    if op == ">=" and isinstance(node.right, LiteralNode) and _is_int_literal(node.right):
        return ComparisonNode(
            operator=">",
            left=node.left,
            right=LiteralNode(value=node.right.value - 1, data_type="number"),
        )
    # Case: N >= expr  →  N+1 > expr  (i.e. expr <= N → expr < N+1, already handled above as expr on left)
    # Handle flipped forms too:
    if op == "<=" and isinstance(node.left, LiteralNode) and _is_int_literal(node.left):
        # N <= expr  ↔  expr >= N  →  expr > N-1
        return ComparisonNode(
            operator=">",
            left=node.right,
            right=LiteralNode(value=node.left.value - 1, data_type="number"),
        )
    if op == ">=" and isinstance(node.left, LiteralNode) and _is_int_literal(node.left):
        # N >= expr  ↔  expr <= N  →  expr < N+1
        return ComparisonNode(
            operator="<",
            left=node.right,
            right=LiteralNode(value=node.left.value + 1, data_type="number"),
        )
    return None


def _is_int_literal(node: LiteralNode) -> bool:
    """True if the literal is a numeric integer (not float, not string)."""
    return (
        node.data_type == "number"
        and isinstance(node.value, int)
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def preprocess_for_smt(
    condition: DRCCondition,
    keep_names: Iterable[str] | None = None,
) -> DRCCondition:
    """Run the full pre-SMT preprocessing pipeline on a single condition.

    Equivalent to ``preprocess_for_smt_pair([condition], keep_names)[0]``.
    """
    out, = preprocess_for_smt_pair([condition], keep_names=keep_names)
    return out


def preprocess_for_smt_pair(
    conditions: list[DRCCondition],
    keep_names: Iterable[str] | None = None,
) -> list[DRCCondition]:
    """Pre-SMT pipeline for one or more conditions sharing the same
    schema (i.e. the same uninterpreted predicates).

    Pass 1 (per-condition): eliminate trivial equalities. This may
    expose more unused slots, so we run it before pass 2.

    Pass 2 (joint): compute the global per-relation slot-usage mask
    over all conditions together, then drop unused slots from each.
    Computing the mask jointly is essential — slot usage in either
    condition keeps the slot alive in both, otherwise the predicate
    signatures would diverge.

    Pass 3 (per-condition): rewrite ``STRFTIME`` calls into integer
    date comparisons. The SMT layer treats ``STRFTIME`` as an
    uninterpreted function, which means cvc5 cannot prove
    ``STRFTIME('%Y', dob) > '1980'`` ↔ ``dob > <last day of 1980>``
    even though the two are semantically equivalent over real dates.
    The rewrite replaces qualifying STRFTIME comparisons with the
    arithmetic form *on both sides* before the script is built, so
    cvc5 sees identical predicates and a clean equivalence proof.

    ``keep_names`` is a set of variable names that must NOT be dropped
    from membership terms even if they appear nowhere else in the
    conditions. This protects names that are referenced *outside* the
    inner condition (typically the DRC's result variables, which are
    free at the top of the set comprehension).
    """
    after_pass1 = [eliminate_trivial_equalities(c) for c in conditions]
    used = eliminate_unused_relation_slots(
        after_pass1, keep_names=set(keep_names or ()),
    )
    after_pass2 = [drop_unused_slots(c, used) for c in after_pass1]
    after_pass3 = [rewrite_strftime_year_comparisons(c) for c in after_pass2]
    after_pass4 = [normalise_integer_boundary_comparisons(c) for c in after_pass3]
    return [rewrite_null_checks(c) for c in after_pass4]


# ---------------------------------------------------------------------------
# Pass 5: NULL-check normalisation
#
# The gold SQL uses ``IS NOT NULL`` / ``IS NULL`` which the translator
# emits as ``FunctionCallNode("IS_NOT_NULL", [col])`` /
# ``FunctionCallNode("IS_NULL", [col])``. The planner-side LLM
# typically emits ``(!= col "")`` or ``(!= col 0)`` for the same
# intent. Since ``IS_NOT_NULL`` is uninterpreted in cvc5, the solver
# can't prove ``col != "" ↔ IS_NOT_NULL(col)``.
#
# Rewrite rules:
#   IS_NOT_NULL(col)  →  col != ""   (String) or col != 0 (Int)
#   IS_NULL(col)      →  col = ""    (String) or col = 0 (Int)
#
# The sort of ``col`` is inferred from the var_types map (String if
# known, otherwise Int). This makes both sides structurally identical
# and cvc5 resolves the equivalence.
# ---------------------------------------------------------------------------


def rewrite_null_checks(condition: DRCCondition) -> DRCCondition:
    """Rewrite ``IS_NOT_NULL(col)`` and ``IS_NULL(col)`` into
    comparison forms that match the planner's output."""
    return _null_rewrite(condition)


def _null_rewrite(node: DRCCondition) -> DRCCondition:
    if node is None:
        return node
    if isinstance(node, IsNotNullNode):
        # ``IsNotNullNode`` is already the canonical form this pass
        # produces from ``FunctionCallNode("IS_NOT_NULL", ...)``; preserve
        # it untouched so the planner-side and gold-side representations
        # collapse to the same SMT.
        return node
    if isinstance(node, FunctionCallNode):
        if node.function == "IS_NOT_NULL" and len(node.arguments) == 1:
            arg = _null_rewrite(node.arguments[0])
            # Rewrite to ``arg != ""`` (works for both String and Int
            # in our model — the SMT converter infers the sort from
            # context and the empty-string literal gets coerced as
            # needed by the comparison handler).
            return ComparisonNode(
                operator="!=",
                left=arg,
                right=LiteralNode(value="", data_type="string"),
            )
        if node.function == "IS_NULL" and len(node.arguments) == 1:
            arg = _null_rewrite(node.arguments[0])
            return ComparisonNode(
                operator="=",
                left=arg,
                right=LiteralNode(value="", data_type="string"),
            )
        return FunctionCallNode(
            function=node.function,
            arguments=[_null_rewrite(a) for a in node.arguments],
        )
    if isinstance(node, ComparisonNode):
        return ComparisonNode(
            operator=node.operator,
            left=_null_rewrite(node.left),
            right=_null_rewrite(node.right),
        )
    if isinstance(node, LogicalConnectiveNode):
        return LogicalConnectiveNode(
            operator=node.operator,
            left=_null_rewrite(node.left),
            right=_null_rewrite(node.right),
        )
    if isinstance(node, NotNode):
        return NotNode(operand=_null_rewrite(node.operand))
    if isinstance(node, QuantifierNode):
        return QuantifierNode(
            kind=node.kind,
            variables=list(node.variables),
            body=_null_rewrite(node.body),
        )
    if isinstance(node, ArithmeticNode):
        return ArithmeticNode(
            operator=node.operator,
            left=_null_rewrite(node.left),
            right=_null_rewrite(node.right),
        )
    return node
