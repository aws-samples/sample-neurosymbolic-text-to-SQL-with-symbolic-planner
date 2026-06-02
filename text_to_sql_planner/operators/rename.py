"""Rename (ρ) operator: relabels one or more columns of a relation.

Produces a relation with the same rows as the input but with renamed
result variables. The condition tree is rewritten so every free
reference to an old column name becomes the corresponding new name,
without disturbing names that have been re-bound by an inner
quantifier (capture-avoiding).

The operator is the standard relational-algebra ``ρ_{a ← b, ...}(R)``.
Its main use cases:

* Disambiguating self-joins. ``Performance_Reviews ⋈
  ρ_{review_id ← r2}(Performance_Reviews)`` is the canonical way to
  produce pairs of distinct reviews for the same employee. Without
  rename the planner had to use cartesian product + selection.
* Aligning column names across union-compatible relations before
  set operations (union, difference). Although difference's own
  positional rename handles this case, an explicit rename in the
  tree makes the intent visible.

Soundness: rename is purely a schema rewrite. The output's row set
equals the input's row set; only the per-column identifier changes.
The capture-avoiding rename ensures no free reference to a renamed
column escapes; quantifier-bound names that happen to coincide with
a renamed column are *not* touched (they're shadowed at that point
in the tree).

Constraints:

* Every key in ``mapping`` must be a column of the input relation
  (a name in ``input.result_variables``).
* No key may map to ``""``.
* No two keys may map to the same new name.
* No new name may collide with an input column that isn't itself
  being renamed away in the same operation.
* Aggregate result variables (``AggregateVariable``) are not
  renamed — the planner only ever projects to aggregates after all
  the structural work is done, so the operator refuses if the input
  relation has aggregate result variables. Callers should put the
  rename before the aggregate projection.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ColumnVariable,
    DRCExpression,
    ResultVariable,
)
from text_to_sql_planner.operators._rename import rename_free
from text_to_sql_planner.types.operators import (
    OperatorFailure,
    OperatorResult,
    OperatorSuccess,
    RenameParams,
)


def _result_variable_names(expr: DRCExpression) -> list[str]:
    """Return the per-result-variable name list for ``expr``."""
    out: list[str] = []
    for rv in expr.result_variables:
        if isinstance(rv, ColumnVariable):
            out.append(rv.name)
        elif isinstance(rv, AggregateVariable):
            out.append(rv.column)
        else:
            out.append("")
    return out


def apply_rename(
    params: RenameParams, inputs: list[DRCExpression],
) -> OperatorResult:
    """Apply rename ``ρ_{mapping}(R)``.

    Returns a new ``DRCExpression`` whose result-variable list has
    each ``mapping[old]`` substituted for ``old``, and whose condition
    has every free reference to ``old`` rewritten to the same new
    name. Other names pass through unchanged.

    Mapping validation:

    * ``mapping`` must be non-empty (an empty rename is a no-op; we
      reject it to flag misuse rather than silently identity-pass).
    * Every key must be the name of an input column.
    * Every value must be a non-empty string.
    * Values must be pairwise distinct.
    * No value may collide with an input column that isn't itself
      a key of ``mapping``.
    * Input relation must have only ``ColumnVariable`` result
      variables (aggregates can't be renamed in this operator —
      project past them first).
    """
    if len(inputs) != 1:
        return OperatorFailure(
            error="Rename requires exactly 1 input relation",
        )

    relation = inputs[0]
    mapping = dict(params.mapping)

    if not mapping:
        return OperatorFailure(
            error="Rename requires a non-empty mapping",
        )

    # Validate input result variables.
    for rv in relation.result_variables:
        if not isinstance(rv, ColumnVariable):
            return OperatorFailure(
                error=(
                    "Rename only supports plain (non-aggregate) result "
                    "variables. Project the aggregate after renaming, "
                    "not before."
                ),
            )

    input_columns = _result_variable_names(relation)
    input_set = set(input_columns)

    # Validate keys.
    for old in mapping:
        if old not in input_set:
            return OperatorFailure(
                error=(
                    f"Rename column '{old}' not found in input relation. "
                    f"Available columns: {input_columns}"
                ),
            )

    # Validate values.
    new_names = list(mapping.values())
    for new in new_names:
        if not isinstance(new, str) or not new.strip():
            return OperatorFailure(
                error=f"Rename target name must be a non-empty string, got {new!r}",
            )
    if len(set(new_names)) != len(new_names):
        return OperatorFailure(
            error=f"Rename targets must be distinct, got {new_names}",
        )

    # No new name may collide with an input column that's not being
    # renamed away. (A name that's both a key and a value, e.g.
    # ``a → b`` and ``b → a``, is fine — the operator does a
    # simultaneous substitution.)
    surviving_input = input_set - set(mapping.keys())
    for new in new_names:
        if new in surviving_input:
            return OperatorFailure(
                error=(
                    f"Rename target '{new}' collides with an input column "
                    f"that isn't being renamed in the same operation"
                ),
            )

    # Rewrite the condition. ``rename_free`` is capture-avoiding so
    # quantifier-shadowed names are left alone. ``mapping`` is applied
    # simultaneously, so swap-style renames (``a → b, b → a``) work
    # correctly.
    new_condition = rename_free(relation.condition, mapping)

    # Rewrite the result variables.
    new_result_vars: list[ResultVariable] = []
    for rv in relation.result_variables:
        # We've already checked above that every result variable is a
        # ColumnVariable.
        if isinstance(rv, ColumnVariable):
            new_result_vars.append(
                ColumnVariable(name=mapping.get(rv.name, rv.name)),
            )
        else:
            new_result_vars.append(rv)

    return OperatorSuccess(
        output=DRCExpression(
            result_variables=new_result_vars,
            condition=new_condition,
        ),
    )
