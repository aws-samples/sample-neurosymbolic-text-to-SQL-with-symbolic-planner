"""Aggregate operator: promote a column result variable to an aggregate.

Takes a relation whose result variables include the column named in
``params.column`` and returns a relation where that one variable is
promoted to a matching :class:`AggregateVariable`. Any other result
variables are preserved unchanged — they become the *group keys* of
the resulting aggregation. The input's condition is passed through
verbatim; only the result-variable list's shape changes.

Two operating modes
-------------------

1. **Single-aggregate-no-keys** (the original case): the input has
   exactly one column-typed result variable, and that's the column
   being aggregated. Output result variables: ``[AggregateVariable]``.
   Equivalent to ``SELECT F(c) FROM …`` with no ``GROUP BY``.

2. **Group-by aggregation** (the dev_78 case): the input has more
   than one column-typed result variable. One of them is being
   aggregated; the others are group keys. Output result variables:
   ``[k1, …, kn, AggregateVariable]`` where ``k1..kn`` are the
   non-aggregated columns in their original order and the aggregate
   replaces the column ``params.column``. The SQL converter's
   :meth:`SQLConverter._generate_with_result_vars` already emits
   ``GROUP BY {k1, …, kn}`` when given this shape, so no separate
   plumbing is needed.

Why this is its own operator
----------------------------

Relational algebra's ``rename`` operator can only relabel a
column-typed result variable as another column name. It cannot
change the result variable's *kind*. Without an explicit aggregate
operator, the planner could build the correct underlying relation
but had no way to transition to the target's aggregate result
variable — every further attempt would produce structurally
identical DRC and get de-duplicated, until the iteration cap fired.
dev_585 was the canonical example of the no-keys version of that
failure mode; dev_78 is the group-by version.

Validation
----------

* The input must have at least one result variable.
* All input result variables must be ``ColumnVariable``s (you can't
  re-aggregate an already-aggregated column, and group keys must
  themselves be plain columns).
* The column named in ``params.column`` must appear in the input's
  result-variable list.
* ``params.function`` must be one of the five SQL aggregates.

The condition is preserved unchanged. Semantically the new
relation's set-theoretic content is the per-group aggregation under
``function``; for the equivalence check, the relation's shape (its
result variables' kinds, functions, and bound columns) encodes that
directly, so the SMT layer doesn't need to be taught about
aggregates as a new construct.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ColumnVariable,
    DRCExpression,
    ResultVariable,
)
from text_to_sql_planner.types.operators import (
    AggregateParams,
    OperatorFailure,
    OperatorResult,
    OperatorSuccess,
)


_AGGREGATE_FUNCS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


def apply_aggregate(
    params: AggregateParams, inputs: list[DRCExpression]
) -> OperatorResult:
    """Apply the aggregate operator.

    Promotes the result variable named in ``params.column`` to an
    :class:`AggregateVariable`. When the input has additional
    column-typed result variables, those become group keys and the
    output preserves them in their original order. See the module
    docstring for full semantics.

    Returns an :class:`OperatorFailure` when validation rejects the
    application so the planner can retry with different inputs /
    params.
    """

    if len(inputs) != 1:
        return OperatorFailure(
            error="Aggregate requires exactly 1 input relation"
        )

    relation = inputs[0]

    if params.function not in _AGGREGATE_FUNCS:
        return OperatorFailure(
            error=(
                f"Aggregate function '{params.function}' is not supported. "
                f"Allowed: {sorted(_AGGREGATE_FUNCS)}"
            )
        )

    if not params.column or not params.column.strip():
        return OperatorFailure(
            error="Aggregate requires a non-empty 'column' parameter"
        )

    if len(relation.result_variables) < 1:
        return OperatorFailure(
            error=(
                "Aggregate requires the input relation to have at least "
                "one result variable; project first to expose the "
                "column being aggregated."
            )
        )

    # Every input result variable must be a column — both the one
    # being aggregated AND the group keys (if any). Re-aggregating
    # an existing aggregate is rejected: SQL doesn't allow nested
    # aggregates without a wrapping subquery, and our DRC shape
    # ``(F (G col))`` is not a thing the equivalence checker
    # recognises.
    for i, rv in enumerate(relation.result_variables):
        if not isinstance(rv, ColumnVariable):
            return OperatorFailure(
                error=(
                    "Aggregate requires every input result variable to "
                    "be a column (group keys must be plain columns; the "
                    "aggregated column must not already be an "
                    f"aggregate). Position {i} has: {type(rv).__name__}"
                )
            )

    # The column being aggregated must appear in the input's result
    # variables. The other columns (if any) become group keys.
    column_names = [rv.name for rv in relation.result_variables]
    if params.column not in column_names:
        return OperatorFailure(
            error=(
                f"Aggregate column '{params.column}' is not in the input "
                f"relation's result variables {column_names}. Project "
                "first so the column appears as a result variable."
            )
        )

    # Build the output result-variable list: every input column stays
    # in its original position, except the named aggregation column,
    # which is promoted to an AggregateVariable.
    output_variables: list[ResultVariable] = []
    for rv in relation.result_variables:
        if isinstance(rv, ColumnVariable) and rv.name == params.column:
            output_variables.append(
                AggregateVariable(
                    function=params.function, column=params.column
                )
            )
        else:
            output_variables.append(rv)

    output = DRCExpression(
        result_variables=output_variables,
        condition=relation.condition,
    )
    return OperatorSuccess(output=output)
