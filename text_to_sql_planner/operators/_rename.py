"""Capture-avoiding free-variable rename, shared by operators that
need it.

The relational-algebra layer's positional rename (used by difference,
union, and the new rename operator) needs to walk a DRC condition
tree and replace free references to a name with a new name, without
clobbering names that have been re-bound by an inner quantifier.
That logic lives here so multiple operators share a single vetted
implementation.

A variable is "free" w.r.t. a sub-tree if it is not bound by an
enclosing quantifier. We thread a ``shadowed`` frozenset of bound
names through the recursion and refuse to rewrite any name that's in
the set at that point. Quantifier-introduced variable lists in
``QuantifierNode`` are NEVER rewritten; their textual identity is
the binding, so renaming them changes which variables the body
references.

Mapping conflict avoidance: when the user passes a mapping that
introduces a name collision (e.g. rename ``a → b`` when ``b`` is
already a free variable in the same sub-tree), the caller is
responsible for choosing a fresh name. This module won't allocate
fresh names — that's a higher-level concern.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ArithmeticNode,
    ComparisonNode,
    DRCCondition,
    FunctionCallNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)


def rename_free(
    condition: DRCCondition, mapping: dict[str, str],
) -> DRCCondition:
    """Rename free variable references in ``condition`` per ``mapping``.

    ``mapping`` maps old names to new names. Names not in the mapping
    are left alone. Names that are bound by an enclosing
    ``QuantifierNode`` (i.e. shadowed) are also left alone — the
    rename only touches *free* references.

    Membership-node slot identifiers and ``VariableRefNode`` leaves
    both participate. Literal nodes and unknown-shape leaves pass
    through unchanged.

    Returns a fresh tree (the input is never mutated). When ``mapping``
    is empty, returns the input unchanged.
    """
    if not mapping:
        return condition
    return _rename_impl(condition, mapping, frozenset())


def _rename_impl(
    condition: DRCCondition,
    mapping: dict[str, str],
    shadowed: frozenset[str],
) -> DRCCondition:
    if condition is None:
        return condition

    if isinstance(condition, VariableRefNode):
        if condition.name in shadowed:
            return condition
        return VariableRefNode(name=mapping.get(condition.name, condition.name))

    if isinstance(condition, MembershipNode):
        new_vars = [
            v if v in shadowed else mapping.get(v, v)
            for v in condition.variables
        ]
        return MembershipNode(variables=new_vars, relation=condition.relation)

    if isinstance(condition, QuantifierNode):
        new_shadowed = shadowed | set(condition.variables)
        return QuantifierNode(
            kind=condition.kind,
            variables=list(condition.variables),
            body=_rename_impl(condition.body, mapping, new_shadowed),
        )

    if isinstance(condition, LogicalConnectiveNode):
        return LogicalConnectiveNode(
            operator=condition.operator,
            left=_rename_impl(condition.left, mapping, shadowed),
            right=_rename_impl(condition.right, mapping, shadowed),
        )

    if isinstance(condition, NotNode):
        return NotNode(operand=_rename_impl(condition.operand, mapping, shadowed))

    if isinstance(condition, ComparisonNode):
        return ComparisonNode(
            operator=condition.operator,
            left=_rename_impl(condition.left, mapping, shadowed),
            right=_rename_impl(condition.right, mapping, shadowed),
        )

    if isinstance(condition, ArithmeticNode):
        return ArithmeticNode(
            operator=condition.operator,
            left=_rename_impl(condition.left, mapping, shadowed),
            right=_rename_impl(condition.right, mapping, shadowed),
        )

    if isinstance(condition, FunctionCallNode):
        return FunctionCallNode(
            function=condition.function,
            arguments=[
                _rename_impl(arg, mapping, shadowed)
                for arg in condition.arguments
            ],
        )

    return condition
