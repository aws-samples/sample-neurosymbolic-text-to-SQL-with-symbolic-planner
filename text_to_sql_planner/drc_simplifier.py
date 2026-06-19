"""DRC expression simplifier: applies correctness-preserving rewrites.

The simplifier is invoked by the planner after each relational-operator
application. Its job is to canonicalize the operator's output so that:

- structurally-different but semantically-identical expressions collapse
  to the same form (helps duplicate detection),
- existentials have minimal scope and bind only variables that are
  actually used (helps cvc5's quantifier instantiation),
- the LLM and equivalence checker see a clean canonical DRC, not the
  raw mechanical output of the operator layer.

The pipeline runs to a fixed point of the printed-Lisp form so multiple
passes can compose in any order. Each pass is a pure function
``(DRCCondition) -> DRCCondition`` that returns the same node when it
doesn't apply.

Default passes (run unconditionally):

1. ``_pass_reflexive_comparison``
       ``(= x x) → True`` and friends.

2. ``_pass_merge_nested_quantifiers``
       ``∃ X. ∃ Y. φ  →  ∃ X,Y. φ`` (and the same for ``∀``).

3. ``_pass_eliminate_trivial_equalities``
       ``∃ v, ... . φ ∧ (= v expr)  →  ∃ ... . φ[v := expr]``
       when ``expr`` does not reference ``v`` (no self-binding) and the
       equality is reachable through top-level conjunctions only (we
       don't rewrite under negation, disjunction, or implication).
       When two bound variables are unified ``(= u v)`` we keep the
       lexicographically-smaller name and drop the other.

4. ``_pass_boolean_simplify``
       Local boolean rewrites:
       ``not (not φ) → φ``,
       ``and φ True → φ``,
       ``and φ φ → φ``,
       ``or φ False → φ``,
       ``or φ φ → φ``,
       ``implies True φ → φ``,
       ``implies False φ → True``.

5. ``_pass_push_existentials_inward``
       ``∃ x. (φ ∧ ψ)  →  φ ∧ (∃ x. ψ)`` when ``x`` is not free in
       ``φ``. This minimises each existential's scope, which improves
       SMT solver performance dramatically because cvc5's E-matching
       only has to instantiate over the smallest body that mentions
       ``x``.

6. ``_pass_drop_unused_binders``
       ``∃ x. φ`` where ``x`` is not free in ``φ`` → ``φ``.
       Also strips per-variable from ``∃ X. φ`` lists.

7. ``_pass_canonical_and_or``
       Flattens ``and``/``or`` chains, sorts operands by structural
       signature, deduplicates structurally-equal operands.

Opt-in pass (NOT run by default):

0. ``_pass_negation_normal_form``
       Pushes ``¬`` to the leaves: ``¬¬φ → φ``, ``¬(φ ∧ ψ) → ¬φ ∨ ¬ψ``,
       ``¬∃X.φ → ∀X.¬φ``, ``¬(a = b) → (a != b)``, etc.
       Run by passing ``simplify_drc(expr, normalize_negation=True)``.

Why NNF is opt-in: it produces wide flat ``∀¬…`` disjunctions that
are *worse* for cvc5's E-matching than the original ``¬∃…`` shape,
because cvc5's quantifier-instantiation tactics handle ``∃`` more
robustly than ``∀``. The planner therefore stores expressions in
their natural ``¬∃`` form, and the equivalence checker / SMT
preprocessor see them in that form. NNF stays available for callers
that explicitly want a canonical form for offline deduplication.

After the rewrite passes settle, every quantifier-bound variable is
renamed to ``_v0, _v1, …`` in left-to-right traversal order. This
alpha-canonicalisation makes structurally-identical-modulo-bound-name
formulas collapse to literally identical trees.

Soundness: every rewrite preserves the truth value of the formula
under every interpretation. The trivial-equality elimination is a
standard one-point rule from first-order logic, the boolean rewrites
are tautologies, existential pushdown is a well-known scoping
identity (``∃x. φ ∧ ψ`` is equivalent to ``φ ∧ ∃x. ψ`` when ``x ∉
FV(φ)``), De Morgan and quantifier-negation are classical FOL
identities, and alpha-renaming is the standard rule of bound-variable
renaming.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from text_to_sql_planner.types.drc import (
    ArithmeticNode,
    ComparisonNode,
    DRCCondition,
    DRCExpression,
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
# Public entry points
# ---------------------------------------------------------------------------


def simplify_drc(
    expr: DRCExpression,
    *,
    normalize_negation: bool = False,
) -> DRCExpression:
    """Simplify a DRC expression to a fixed point of the rewrite pipeline,
    then alpha-canonicalise bound variables to a deterministic sequence.

    The condition rewrites run first to a fixed point. After they
    settle, every quantifier-bound variable is renamed to ``_v0,
    _v1, ...`` in left-to-right traversal order. This last step makes
    structurally-identical-modulo-bound-variable-names formulas reduce
    to literally identical trees, which is what powers duplicate
    detection in the planner and gives cvc5's E-matching consistent
    Skolem terms across both sides of an equivalence check.

    ``normalize_negation`` controls whether the negation-normal-form
    pass (``¬∃ → ∀¬``, ``¬∀ → ∃¬``, De Morgan, comparison flips) runs.
    The default is ``False``: NNF produces wide flat disjunctions of
    ``∀¬…`` clauses that are harder for cvc5's E-matching to align
    against ``∃`` formulas in the typical "≥N − ≥M" pattern. Callers
    that explicitly want a uniform NNF-canonical form (e.g. for
    structural deduplication of equivalence classes the other passes
    don't reach) can set this flag.
    """
    if normalize_negation:
        passes = _PASSES_WITH_NNF
    else:
        passes = _PASSES_NO_NNF
    simplified = DRCExpression(
        result_variables=expr.result_variables,
        condition=_simplify_condition(expr.condition, passes),
    )
    return _alpha_canonicalize_expression(simplified)


def _simplify_condition(
    node: DRCCondition,
    passes: list[tuple[str, "Callable[[DRCCondition], DRCCondition]"]],
) -> DRCCondition:
    """Run every rewrite pass until printed-Lisp output stabilises.

    Each iteration walks the tree once per pass; with N passes and a
    bounded fixed-point loop this is O(N · iterations · tree_size)
    in the worst case. In practice expressions converge in 2–4 outer
    iterations.
    """
    if node is None:
        return node
    prev_signature: object = None
    for _ in range(_MAX_FIXED_POINT_ITERATIONS):
        for _name, rule in passes:
            node = _walk(node, rule)
        signature = _signature(node)
        if signature == prev_signature:
            break
        prev_signature = signature
    return node


_MAX_FIXED_POINT_ITERATIONS = 12


# ---------------------------------------------------------------------------
# Per-pass implementations
# ---------------------------------------------------------------------------


def _pass_merge_nested_quantifiers(node: DRCCondition) -> DRCCondition:
    """``∃ X. ∃ Y. φ → ∃ X,Y. φ`` (same for ``∀``).

    Only merges when the kinds match and the body is itself a
    quantifier of the same kind.
    """
    if not isinstance(node, QuantifierNode):
        return node
    merged_vars = list(node.variables)
    current = node.body
    while (
        isinstance(current, QuantifierNode)
        and current.kind == node.kind
    ):
        merged_vars.extend(current.variables)
        current = current.body
    if current is node.body:
        return node
    return QuantifierNode(kind=node.kind, variables=merged_vars, body=current)


def _pass_eliminate_trivial_equalities(node: DRCCondition) -> DRCCondition:
    """``∃ v, ... . φ ∧ (= v expr) → ∃ ... . φ[v := expr]``.

    Iterates inside the binder list until no more equalities pin a
    bound name. Recognises both orderings ``(= v expr)`` and
    ``(= expr v)``.

    Soundness rule: substitute only when the substituent is a
    variable reference (``_substitute`` rewrites both ``VariableRefNode``
    leaves and ``MembershipNode`` slot identifiers in that case), or
    when the bound variable does not appear in any ``MembershipNode``
    slot of the body (in which case there's nothing to substitute into
    a slot anyway). DRC has no representation for "literal at slot k",
    so we must NOT drop the binder when the variable would leave a
    dangling membership-slot reference behind.

    When two bound names are unified ``(= u v)`` we keep the
    lexicographically-smaller name.
    """
    if not isinstance(node, QuantifierNode) or node.kind != "exists":
        return node

    bound = list(node.variables)
    body = node.body

    changed = True
    while changed:
        changed = False
        bound_set = set(bound)
        for var, expr in _collect_top_level_equalities(body):
            if var not in bound_set:
                continue
            if _references(expr, var):
                continue

            substituent_is_variable = isinstance(expr, VariableRefNode)
            if not substituent_is_variable and _appears_in_membership(body, var):
                # Literal substituent + membership-slot use → unsound to
                # drop the binder. Skip this equality.
                continue

            if substituent_is_variable and expr.name in bound_set:
                # Unify two bound names: keep the smaller one.
                keep, drop = sorted([var, expr.name])
                body = _drop_equality(body, drop, VariableRefNode(name=keep))
                body = _substitute(body, {drop: VariableRefNode(name=keep)})
                bound = [v for v in bound if v != drop]
                changed = True
                break

            body = _drop_equality(body, var, expr)
            body = _substitute(body, {var: expr})
            bound = [v for v in bound if v != var]
            changed = True
            break

    if bound == list(node.variables) and _structural_equal(body, node.body):
        return node
    if not bound:
        return body
    return QuantifierNode(kind="exists", variables=bound, body=body)


def _pass_boolean_simplify(node: DRCCondition) -> DRCCondition:
    """Local boolean rewrites that drop tautologies and idempotent
    duplicates.

    The ``True`` and ``False`` constants here are the placeholder
    ``LiteralNode``\\s emitted by other passes (e.g. when an equality
    has been consumed). Idempotence is recognised by structural
    equality on subterms.
    """
    if isinstance(node, NotNode):
        if isinstance(node.operand, NotNode):
            return node.operand.operand  # ¬¬φ → φ
        if _is_true(node.operand):
            return _FALSE
        if _is_false(node.operand):
            return _TRUE
        return node

    if isinstance(node, LogicalConnectiveNode):
        left, right = node.left, node.right
        op = node.operator

        if op == "and":
            if _is_true(left):
                return right
            if _is_true(right):
                return left
            if _is_false(left) or _is_false(right):
                return _FALSE
            if _structural_equal(left, right):
                return left
            return node

        if op == "or":
            if _is_false(left):
                return right
            if _is_false(right):
                return left
            if _is_true(left) or _is_true(right):
                return _TRUE
            if _structural_equal(left, right):
                return left
            return node

        if op == "implies":
            if _is_true(left):
                return right       # True → φ ≡ φ
            if _is_false(left):
                return _TRUE       # False → φ ≡ True
            if _is_true(right):
                return _TRUE       # φ → True ≡ True
            if _structural_equal(left, right):
                return _TRUE       # φ → φ ≡ True
            return node

    return node


def _pass_push_existentials_inward(node: DRCCondition) -> DRCCondition:
    """``∃ X. (φ ∧ ψ) → (φ ∧ ∃ Y. ψ)`` when the binders in ``X`` that
    don't appear in ``φ`` can be pushed into ``ψ`` alone.

    Concretely: split ``X`` into ``X_φ`` (names free in ``φ``) and
    ``X_ψ`` (names not free in ``φ``). Move ``X_ψ`` into ``ψ``. If
    after that the outer existential has no remaining binders, drop
    it.
    """
    if not isinstance(node, QuantifierNode) or node.kind != "exists":
        return node
    if not isinstance(node.body, LogicalConnectiveNode) or node.body.operator != "and":
        return node

    left, right = node.body.left, node.body.right

    # Decide for each binder whether it's free in left or right (or
    # both). We have three buckets: only-in-left, only-in-right, in-both.
    only_left: list[str] = []
    only_right: list[str] = []
    shared: list[str] = []
    for v in node.variables:
        in_left = _references(left, v)
        in_right = _references(right, v)
        if in_left and in_right:
            shared.append(v)
        elif in_left:
            only_left.append(v)
        elif in_right:
            only_right.append(v)
        else:
            # Bound but unused — pass 5 (drop_unused_binders) handles it.
            shared.append(v)

    # If everything is shared or unused, no progress.
    if not only_left and not only_right:
        return node

    new_left: DRCCondition = left
    if only_left:
        new_left = QuantifierNode(
            kind="exists", variables=only_left, body=left,
        )
    new_right: DRCCondition = right
    if only_right:
        new_right = QuantifierNode(
            kind="exists", variables=only_right, body=right,
        )

    inner = LogicalConnectiveNode(operator="and", left=new_left, right=new_right)
    if not shared:
        return inner
    return QuantifierNode(kind="exists", variables=shared, body=inner)


def _pass_drop_unused_binders(node: DRCCondition) -> DRCCondition:
    """``∃ X. φ → ∃ X'. φ`` where ``X' = X ∩ FV(φ)``.

    If ``X' = ∅``, returns ``φ`` directly.
    """
    if not isinstance(node, QuantifierNode):
        return node
    used = [v for v in node.variables if _references(node.body, v)]
    if used == list(node.variables):
        return node
    if not used:
        return node.body
    return QuantifierNode(kind=node.kind, variables=used, body=node.body)


def _pass_reflexive_comparison(node: DRCCondition) -> DRCCondition:
    """Comparisons with structurally-equal operands collapse to constants.

    - ``(= x x)`` → ``True``,  ``(!= x x)`` → ``False``
    - ``(<= x x)`` → ``True``, ``(>= x x)`` → ``True``
    - ``(< x x)`` → ``False``, ``(> x x)`` → ``False``

    Soundness: a comparison with identical operands evaluates to a
    constant under every interpretation. The boolean-simplify pass
    that follows then prunes any resulting ``True``/``False`` from
    surrounding ``and``/``or`` chains.
    """
    if not isinstance(node, ComparisonNode):
        return node
    if not _structural_equal(node.left, node.right):
        return node
    if node.operator in ("=", "<=", ">="):
        return _TRUE
    if node.operator in ("!=", "<", ">"):
        return _FALSE
    return node


def _pass_negation_normal_form(node: DRCCondition) -> DRCCondition:
    """Push ``¬`` toward the leaves (negation-normal form).

    Rewrites at a single ``not`` node:

    - ``¬¬φ → φ``
    - ``¬(φ ∧ ψ) → ¬φ ∨ ¬ψ``
    - ``¬(φ ∨ ψ) → ¬φ ∧ ¬ψ``
    - ``¬(φ → ψ) → φ ∧ ¬ψ``
    - ``¬∃X. φ → ∀X. ¬φ``
    - ``¬∀X. φ → ∃X. ¬φ``
    - ``¬(a = b) → (a != b)`` (and the symmetric flips for the other
      five comparison operators).

    Soundness: each of these is a classical first-order tautology.
    Useful because the existing ``push_existentials_inward`` pass only
    operates on ``∃`` directly under ``and``; getting the negation past
    the quantifier turns a ``¬∃`` into ``∀¬`` and exposes the body for
    further simplification, which is exactly the structure that breaks
    cvc5 in the "exactly N" pattern.
    """
    if not isinstance(node, NotNode):
        return node
    inner = node.operand

    # ¬¬φ → φ
    if isinstance(inner, NotNode):
        return inner.operand

    # ¬(φ ∧ ψ) → ¬φ ∨ ¬ψ;  ¬(φ ∨ ψ) → ¬φ ∧ ¬ψ
    if isinstance(inner, LogicalConnectiveNode):
        if inner.operator == "and":
            return LogicalConnectiveNode(
                operator="or",
                left=NotNode(operand=inner.left),
                right=NotNode(operand=inner.right),
            )
        if inner.operator == "or":
            return LogicalConnectiveNode(
                operator="and",
                left=NotNode(operand=inner.left),
                right=NotNode(operand=inner.right),
            )
        if inner.operator == "implies":
            # ¬(φ → ψ) ≡ φ ∧ ¬ψ
            return LogicalConnectiveNode(
                operator="and",
                left=inner.left,
                right=NotNode(operand=inner.right),
            )

    # ¬∃X. φ → ∀X. ¬φ ; ¬∀X. φ → ∃X. ¬φ
    if isinstance(inner, QuantifierNode):
        flipped_kind = "forall" if inner.kind == "exists" else "exists"
        return QuantifierNode(
            kind=flipped_kind,
            variables=list(inner.variables),
            body=NotNode(operand=inner.body),
        )

    # ¬(a OP b) → (a OP' b)  where OP' is the negated comparison.
    if isinstance(inner, ComparisonNode):
        flip = {"=": "!=", "!=": "=", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}
        new_op = flip.get(inner.operator)
        if new_op is not None:
            return ComparisonNode(
                operator=new_op, left=inner.left, right=inner.right,
            )

    return node


# ---------------------------------------------------------------------------
# Variadic-and/or canonicalisation
# ---------------------------------------------------------------------------


def _pass_canonical_and_or(node: DRCCondition) -> DRCCondition:
    """Treat ``and``/``or`` as variadic: flatten same-operator chains,
    sort children by signature, and remove structurally-equal duplicates.

    Soundness: ``and`` and ``or`` are commutative, associative, and
    idempotent. So ``(and (and a b) c)`` is equivalent to ``(and a b
    c)``, which is equivalent to any permutation, and to the same set
    after de-duplication. The output of this pass has a unique
    canonical shape per equivalence class, which means structurally-
    identical formulas (modulo ordering and grouping) collapse to the
    same tree — the duplicate detector and the LLM both benefit.

    Implementation: at any ``LogicalConnectiveNode`` whose operator is
    ``and`` or ``or``, gather the flat list of operands, sort by
    structural signature, deduplicate, and rebuild a left-associative
    chain in canonical order. Single-operand collapses to that
    operand. Empty (after dedup, impossible from a binary node) would
    collapse to the boolean unit; we don't generate those here.
    """
    if not isinstance(node, LogicalConnectiveNode):
        return node
    if node.operator not in ("and", "or"):
        return node

    operands = _flatten_same_op(node, node.operator)

    # Deduplicate by signature, preserving the *first* occurrence.
    seen: set[tuple] = set()
    deduped: list[DRCCondition] = []
    for op in operands:
        sig = _signature(op)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(op)

    # Sort by signature so commutativity is canonicalised.
    deduped.sort(key=_signature)

    # If nothing changed, return the node unchanged so the fixed-point
    # loop converges.
    if len(deduped) == 2 and _structural_equal(deduped[0], node.left) and _structural_equal(deduped[1], node.right):
        return node

    # Rebuild as a left-associative chain.
    if len(deduped) == 1:
        return deduped[0]
    result = deduped[0]
    for child in deduped[1:]:
        result = LogicalConnectiveNode(
            operator=node.operator, left=result, right=child,
        )
    return result


def _flatten_same_op(node: DRCCondition, op: str) -> list[DRCCondition]:
    """Collect all immediate children of an ``and``/``or`` chain whose
    operator matches ``op``. Stops at any sub-node that is not the
    same operator (so it doesn't descend through nested ``not``,
    ``forall``, etc.)."""
    if isinstance(node, LogicalConnectiveNode) and node.operator == op:
        return _flatten_same_op(node.left, op) + _flatten_same_op(node.right, op)
    return [node]


# ---------------------------------------------------------------------------
# Alpha-canonicalisation
# ---------------------------------------------------------------------------


def _alpha_canonicalize_expression(expr: DRCExpression) -> DRCExpression:
    """Rename every bound (i.e. quantifier-introduced) variable to a
    deterministic sequence ``_v0, _v1, ...`` based on a left-to-right
    traversal order. Preserves free variables (including the result-
    variable names) verbatim.

    Soundness: alpha-renaming is the standard rename-of-bound-variables
    rule from first-order logic; the truth value of the formula is
    unchanged under any consistent rename of bound names that doesn't
    capture free variables. We avoid capture by choosing names that
    are guaranteed not to clash with any free variable in the
    expression.

    Why it's an outer-level pass (operates on ``DRCExpression`` rather
    than ``DRCCondition``): we need to know the result-variable names
    to put them in the "free name" reserve set. Running this on a
    bare condition would risk producing ``_v0`` etc. for binders that
    happen to clash with an outer-scope free name we can't see.
    """
    free_reserve: set[str] = set()
    for rv in expr.result_variables:
        if hasattr(rv, "name"):
            free_reserve.add(rv.name)
        elif hasattr(rv, "column"):
            free_reserve.add(rv.column)

    # Also reserve every free variable that appears anywhere in the
    # condition tree (variables that are referenced but not bound in
    # an enclosing quantifier). Bound names will be replaced with
    # ``_v0, _v1, …``; free names stay untouched.
    free_in_condition: set[str] = set()
    _collect_free_vars(expr.condition, bound_stack=set(), free=free_in_condition)
    free_reserve |= free_in_condition

    counter = [0]

    def fresh() -> str:
        while True:
            candidate = f"_v{counter[0]}"
            counter[0] += 1
            if candidate not in free_reserve:
                return candidate

    new_condition = _alpha_rename_node(
        expr.condition, scope=({}), fresh=fresh,
    )
    if new_condition is expr.condition:
        return expr
    return DRCExpression(
        result_variables=expr.result_variables,
        condition=new_condition,
    )


def _alpha_rename_node(
    node: DRCCondition,
    scope: dict[str, str],
    fresh: Callable[[], str],
) -> DRCCondition:
    """Capture-avoiding alpha rename. ``scope`` maps original bound
    names to their renamed forms; free names stay untouched.
    """
    if node is None:
        return node

    if isinstance(node, VariableRefNode):
        repl = scope.get(node.name)
        if repl is None:
            return node
        return VariableRefNode(name=repl)

    if isinstance(node, MembershipNode):
        new_vars = [scope.get(v, v) for v in node.variables]
        if new_vars == node.variables:
            return node
        return MembershipNode(variables=new_vars, relation=node.relation)

    if isinstance(node, IsNotNullNode):
        # ``IsNotNullNode.column`` is renamed under ``scope`` the same way
        # ``MembershipNode.variables[i]`` is.
        new_col = scope.get(node.column, node.column)
        if new_col == node.column:
            return node
        return IsNotNullNode(column=new_col)

    if isinstance(node, QuantifierNode):
        # Allocate a fresh name for each bound variable, in order. The
        # stable sequence (`_v0, _v1, …`) is what gives canonicalisation.
        new_scope = dict(scope)
        renamed: list[str] = []
        for v in node.variables:
            new_name = fresh()
            new_scope[v] = new_name
            renamed.append(new_name)
        new_body = _alpha_rename_node(node.body, new_scope, fresh)
        return QuantifierNode(
            kind=node.kind,
            variables=renamed,
            body=new_body,
        )

    if isinstance(node, LogicalConnectiveNode):
        new_left = _alpha_rename_node(node.left, scope, fresh)
        new_right = _alpha_rename_node(node.right, scope, fresh)
        if new_left is node.left and new_right is node.right:
            return node
        return LogicalConnectiveNode(
            operator=node.operator, left=new_left, right=new_right,
        )

    if isinstance(node, NotNode):
        new_op = _alpha_rename_node(node.operand, scope, fresh)
        if new_op is node.operand:
            return node
        return NotNode(operand=new_op)

    if isinstance(node, ComparisonNode):
        new_left = _alpha_rename_node(node.left, scope, fresh)
        new_right = _alpha_rename_node(node.right, scope, fresh)
        if new_left is node.left and new_right is node.right:
            return node
        return ComparisonNode(
            operator=node.operator, left=new_left, right=new_right,
        )

    if isinstance(node, ArithmeticNode):
        new_left = _alpha_rename_node(node.left, scope, fresh)
        new_right = _alpha_rename_node(node.right, scope, fresh)
        if new_left is node.left and new_right is node.right:
            return node
        return ArithmeticNode(
            operator=node.operator, left=new_left, right=new_right,
        )

    if isinstance(node, FunctionCallNode):
        new_args = [_alpha_rename_node(a, scope, fresh) for a in node.arguments]
        if all(a is b for a, b in zip(new_args, node.arguments)):
            return node
        return FunctionCallNode(function=node.function, arguments=new_args)

    return node


def _collect_free_vars(
    node: DRCCondition,
    bound_stack: set[str],
    free: set[str],
) -> None:
    """Walk the AST and add to ``free`` every variable name that is
    referenced (via ``VariableRefNode`` or as a ``MembershipNode`` slot)
    and is not currently bound by an enclosing quantifier.
    """
    if node is None:
        return
    if isinstance(node, VariableRefNode):
        if node.name not in bound_stack:
            free.add(node.name)
        return
    if isinstance(node, MembershipNode):
        for v in node.variables:
            if v not in bound_stack:
                free.add(v)
        return
    if isinstance(node, IsNotNullNode):
        # ``IsNotNullNode.column`` is a column-binding name, free unless
        # bound by an enclosing quantifier — same treatment as a
        # ``MembershipNode`` slot.
        if node.column and node.column not in bound_stack:
            free.add(node.column)
        return
    if isinstance(node, QuantifierNode):
        new_bound = bound_stack | set(node.variables)
        _collect_free_vars(node.body, new_bound, free)
        return
    if isinstance(node, LogicalConnectiveNode):
        _collect_free_vars(node.left, bound_stack, free)
        _collect_free_vars(node.right, bound_stack, free)
        return
    if isinstance(node, NotNode):
        _collect_free_vars(node.operand, bound_stack, free)
        return
    if isinstance(node, ComparisonNode):
        _collect_free_vars(node.left, bound_stack, free)
        _collect_free_vars(node.right, bound_stack, free)
        return
    if isinstance(node, ArithmeticNode):
        _collect_free_vars(node.left, bound_stack, free)
        _collect_free_vars(node.right, bound_stack, free)
        return
    if isinstance(node, FunctionCallNode):
        for a in node.arguments:
            _collect_free_vars(a, bound_stack, free)
        return


# Pipeline order matters:
#
# - NNF first (push ¬ to leaves) so subsequent passes see ``∀X. ¬φ`` /
#   ``∃X. ¬φ`` instead of ``¬∃X. φ`` / ``¬∀X. φ``.
# - Reflexive comparisons next, before booleans, so the ``True``/``False``
#   they produce gets folded by the boolean simplifier on the same iteration.
# - Equality elimination produces ``True`` placeholders that the boolean
#   simplifier prunes out of ``and``/``or`` chains.
# - Existential pushdown is most effective once equalities have been
#   folded away, because shrinking the body exposes more free-variable
#   splits.
# - and/or canonicalisation flattens chains and sorts/dedups operands;
#   this is the rule that makes structurally-equivalent formulas reduce
#   to identical trees, which is essential for the duplicate detector
#   and for cvc5's E-matching alignment.
# - Unused-binder drop runs after pushdown so it can remove binders
#   pushdown stranded.
#
# Two pass lists are exposed so callers can opt in or out of NNF:
#
#   ``_PASSES_NO_NNF``      — default, used by the planner's
#                              ``simplify_drc(expr)`` call. Does NOT
#                              push ``¬`` past quantifiers, so the
#                              expression keeps its natural ``¬∃`` shape
#                              that cvc5's E-matching aligns better.
#
#   ``_PASSES_WITH_NNF``    — opt-in via
#                              ``simplify_drc(expr, normalize_negation=True)``.
#                              Produces a strict negation-normal-form
#                              canonical that's useful for offline
#                              canonicalisation but not for the cvc5
#                              equivalence path.
_PASSES_NO_NNF: list[tuple[str, Callable[[DRCCondition], DRCCondition]]] = [
    ("reflexive_comparison", _pass_reflexive_comparison),
    ("merge_nested_quantifiers", _pass_merge_nested_quantifiers),
    ("eliminate_trivial_equalities", _pass_eliminate_trivial_equalities),
    ("boolean_simplify", _pass_boolean_simplify),
    ("push_existentials_inward", _pass_push_existentials_inward),
    ("drop_unused_binders", _pass_drop_unused_binders),
    ("canonical_and_or", _pass_canonical_and_or),
]

_PASSES_WITH_NNF: list[tuple[str, Callable[[DRCCondition], DRCCondition]]] = [
    ("negation_normal_form", _pass_negation_normal_form),
] + _PASSES_NO_NNF


# ---------------------------------------------------------------------------
# Generic walker — applies a per-node rewrite bottom-up
# ---------------------------------------------------------------------------


def _walk(node: DRCCondition, rule: Callable[[DRCCondition], DRCCondition]) -> DRCCondition:
    """Apply ``rule`` to every node bottom-up."""
    if node is None:
        return node

    if isinstance(node, QuantifierNode):
        new_body = _walk(node.body, rule)
        if new_body is not node.body:
            node = QuantifierNode(
                kind=node.kind, variables=list(node.variables), body=new_body,
            )
        return rule(node)

    if isinstance(node, LogicalConnectiveNode):
        new_left = _walk(node.left, rule)
        new_right = _walk(node.right, rule)
        if new_left is not node.left or new_right is not node.right:
            node = LogicalConnectiveNode(
                operator=node.operator, left=new_left, right=new_right,
            )
        return rule(node)

    if isinstance(node, NotNode):
        new_operand = _walk(node.operand, rule)
        if new_operand is not node.operand:
            node = NotNode(operand=new_operand)
        return rule(node)

    if isinstance(node, ComparisonNode):
        new_left = _walk(node.left, rule)
        new_right = _walk(node.right, rule)
        if new_left is not node.left or new_right is not node.right:
            node = ComparisonNode(
                operator=node.operator, left=new_left, right=new_right,
            )
        return rule(node)

    if isinstance(node, ArithmeticNode):
        new_left = _walk(node.left, rule)
        new_right = _walk(node.right, rule)
        if new_left is not node.left or new_right is not node.right:
            node = ArithmeticNode(
                operator=node.operator, left=new_left, right=new_right,
            )
        return rule(node)

    if isinstance(node, FunctionCallNode):
        new_args = [_walk(a, rule) for a in node.arguments]
        if any(a is not b for a, b in zip(new_args, node.arguments)):
            node = FunctionCallNode(function=node.function, arguments=new_args)
        return rule(node)

    if isinstance(node, IsNotNullNode):
        # Leaf node — no condition-tree children to recurse into; the
        # bound-variable-like ``column`` field is a string. Just apply
        # the rule, mirroring the ``MembershipNode`` / ``LiteralNode``
        # leaf-walk path.
        return rule(node)

    # Leaves — no recursion, just apply the rule.
    return rule(node)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Tagged literals used as boolean constants by the rewrite passes.
# We use the integer literals 1 and 0 as True / False placeholders;
# they're only ever introduced by ``_drop_equality`` and consumed by
# ``_pass_boolean_simplify``. Other code paths never compare against
# these specific values.
_TRUE = LiteralNode(value=1, data_type="number")
_FALSE = LiteralNode(value=0, data_type="number")


def _is_true(node: DRCCondition) -> bool:
    return (
        isinstance(node, LiteralNode)
        and node.data_type == "number"
        and node.value == 1
    )


def _is_false(node: DRCCondition) -> bool:
    return (
        isinstance(node, LiteralNode)
        and node.data_type == "number"
        and node.value == 0
    )


def _references(node: DRCCondition, var: str) -> bool:
    """Does ``node`` contain a free reference to ``var``?

    Tracks quantifier shadowing: a name that's rebound inside the
    sub-tree is not counted as a free reference.
    """
    if node is None:
        return False
    if isinstance(node, VariableRefNode):
        return node.name == var
    if isinstance(node, MembershipNode):
        return var in node.variables
    if isinstance(node, IsNotNullNode):
        # ``IsNotNullNode.column`` is a column-binding name treated the
        # same as a ``MembershipNode`` slot.
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


def _appears_in_membership(node: DRCCondition, var: str) -> bool:
    """Does any ``MembershipNode`` in ``node`` use ``var`` as a slot
    identifier (in its ``variables`` list)?

    Used by the equality-elimination pass to decide whether a
    literal-substituent rewrite would leave a dangling slot reference.
    Tracks quantifier shadowing the same way as ``_references``.

    ``IsNotNullNode.column`` is also treated as a positional slot
    identifier here: substituting a literal for a name that appears as
    an ``IsNotNullNode`` column would leave the column name dangling
    the same way it would for a membership slot.
    """
    if node is None:
        return False
    if isinstance(node, MembershipNode):
        return var in node.variables
    if isinstance(node, IsNotNullNode):
        return node.column == var
    if isinstance(node, QuantifierNode):
        if var in node.variables:
            return False
        return _appears_in_membership(node.body, var)
    if isinstance(node, LogicalConnectiveNode):
        return _appears_in_membership(node.left, var) or _appears_in_membership(node.right, var)
    if isinstance(node, NotNode):
        return _appears_in_membership(node.operand, var)
    if isinstance(node, ComparisonNode):
        return _appears_in_membership(node.left, var) or _appears_in_membership(node.right, var)
    if isinstance(node, ArithmeticNode):
        return _appears_in_membership(node.left, var) or _appears_in_membership(node.right, var)
    if isinstance(node, FunctionCallNode):
        return any(_appears_in_membership(a, var) for a in node.arguments)
    return False


def _collect_top_level_equalities(node: DRCCondition) -> list[tuple[str, DRCCondition]]:
    """Return ``(var_name, expr)`` for every ``(= v expr)`` reachable
    through top-level AND chains in ``node``.

    Both orderings ``(= v e)`` and ``(= e v)`` are recognized; if one
    side is a variable reference and the other isn't, that variable is
    returned. If *both* sides are variable references, both
    ``(left, right)`` and ``(right, left)`` are returned so the caller
    can pick whichever side happens to be currently bound.

    We stop at OR / NOT / IMPLIES / quantifier boundaries because
    substituting under those changes semantics.
    """
    out: list[tuple[str, DRCCondition]] = []
    if isinstance(node, ComparisonNode) and node.operator == "=":
        left_is_var = isinstance(node.left, VariableRefNode)
        right_is_var = isinstance(node.right, VariableRefNode)
        if left_is_var and right_is_var:
            # Either side could be the "bound" variable; emit both so
            # the caller can pick whichever fits.
            out.append((node.left.name, node.right))
            out.append((node.right.name, node.left))
        elif left_is_var and not _references(node.right, node.left.name):
            out.append((node.left.name, node.right))
        elif right_is_var and not _references(node.left, node.right.name):
            out.append((node.right.name, node.left))
    elif isinstance(node, LogicalConnectiveNode) and node.operator == "and":
        out.extend(_collect_top_level_equalities(node.left))
        out.extend(_collect_top_level_equalities(node.right))
    return out


def _drop_equality(
    node: DRCCondition, var: str, expr: DRCCondition,
) -> DRCCondition:
    """Replace the matching equality at the top level of an AND chain
    with a ``True`` placeholder; the boolean-simplify pass then
    eliminates it. Returns ``node`` unchanged if no match.
    """
    if isinstance(node, ComparisonNode) and node.operator == "=":
        if (
            isinstance(node.left, VariableRefNode)
            and node.left.name == var
            and _structural_equal(node.right, expr)
        ):
            return _TRUE
        if (
            isinstance(node.right, VariableRefNode)
            and node.right.name == var
            and _structural_equal(node.left, expr)
        ):
            return _TRUE
        return node
    if isinstance(node, LogicalConnectiveNode) and node.operator == "and":
        new_left = _drop_equality(node.left, var, expr)
        if not _structural_equal(new_left, node.left):
            return LogicalConnectiveNode(
                operator="and", left=new_left, right=node.right,
            )
        new_right = _drop_equality(node.right, var, expr)
        if not _structural_equal(new_right, node.right):
            return LogicalConnectiveNode(
                operator="and", left=node.left, right=new_right,
            )
    return node


def _substitute(
    node: DRCCondition, mapping: dict[str, DRCCondition],
) -> DRCCondition:
    """Capture-avoiding substitution of free variables.

    Membership slots accept only identifier substituents — when the
    substituent is a literal (no DRC representation for "literal at
    slot k"), the original name is kept. The equality elimination pass
    in :mod:`text_to_sql_planner.equivalence.smt_preprocessing` works
    even with this fallback because the unused-slot pruner discovers
    those slots are dead and drops them.
    """
    if node is None:
        return node
    if isinstance(node, VariableRefNode):
        repl = mapping.get(node.name)
        return repl if repl is not None else node
    if isinstance(node, MembershipNode):
        new_vars: list[str] = []
        rewrote = False
        for v in node.variables:
            repl = mapping.get(v)
            if isinstance(repl, VariableRefNode):
                new_vars.append(repl.name)
                rewrote = True
            else:
                new_vars.append(v)
        if rewrote:
            return MembershipNode(variables=new_vars, relation=node.relation)
        return node
    if isinstance(node, IsNotNullNode):
        # ``IsNotNullNode.column`` is a column-binding name handled the
        # same way as a ``MembershipNode`` slot: only variable-to-variable
        # substitutions apply (a literal substituent has no representation
        # as a column name, so the original is kept).
        repl = mapping.get(node.column)
        if isinstance(repl, VariableRefNode):
            return IsNotNullNode(column=repl.name)
        return node
    if isinstance(node, QuantifierNode):
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


def _structural_equal(a: DRCCondition, b: DRCCondition) -> bool:
    """Structural equality on DRC nodes."""
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


def _signature(node: DRCCondition) -> tuple:
    """Compact hashable representation of a DRC tree, used to detect
    fixed-point convergence. Cheaper than printing to Lisp."""
    if node is None:
        return ("none",)
    if isinstance(node, VariableRefNode):
        return ("var", node.name)
    if isinstance(node, LiteralNode):
        return ("lit", node.data_type, node.value)
    if isinstance(node, MembershipNode):
        return ("in", node.relation, tuple(node.variables))
    if isinstance(node, IsNotNullNode):
        return ("isnotnull", node.column)
    if isinstance(node, QuantifierNode):
        return ("q", node.kind, tuple(node.variables), _signature(node.body))
    if isinstance(node, LogicalConnectiveNode):
        return ("log", node.operator, _signature(node.left), _signature(node.right))
    if isinstance(node, NotNode):
        return ("not", _signature(node.operand))
    if isinstance(node, ComparisonNode):
        return ("cmp", node.operator, _signature(node.left), _signature(node.right))
    if isinstance(node, ArithmeticNode):
        return ("arith", node.operator, _signature(node.left), _signature(node.right))
    if isinstance(node, FunctionCallNode):
        return ("fn", node.function, tuple(_signature(a) for a in node.arguments))
    return ("unknown",)
