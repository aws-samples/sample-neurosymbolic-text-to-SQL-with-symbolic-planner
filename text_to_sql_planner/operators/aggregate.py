"""Aggregate operator: promote a column result variable to an aggregate.

Takes a relation with a single ``ColumnVariable`` result variable and
returns a relation whose single result variable is the matching
:class:`AggregateVariable`. The input's condition is preserved
verbatim — only the result variable's *kind* changes.

Why this is its own operator
----------------------------

Relational algebra's ``rename`` operator can only relabel a
column-typed result variable as another column name. It cannot
change the result variable's *kind*. Without an explicit aggregate
operator, the planner could build the correct underlying relation
(e.g. ``{BountyAmount | …}``) but had no way to transition to the
target ``{(SUM BountyAmount) | …}`` — every further attempt would
produce structurally-identical DRC and get de-duplicated, until the
iteration cap fired. dev_585 is the canonical example of that
failure mode.

This operator addresses the simpler "aggregate the whole relation
to a single value" case. Group-by aggregation continues to be
expressed as ``projection`` on a join with a mixed column /
aggregate column list, exactly as it is today.

Validation
----------

* The input must have *exactly one* result variable.
* That variable must be a ``ColumnVariable`` (you can't aggregate
  an already-aggregate column).
* The variable's name must equal ``params.column`` so the operator
  can't be misapplied to a column the input doesn't have.
* ``params.function`` must be one of the five SQL aggregates the
  framework supports.

The condition is passed through unchanged. Semantically the new
relation's set-theoretic content is "the aggregate of the input set
under ``function``" — but for the equivalence check, the relation's
*shape* (its result variable's kind, function, and bound column)
already encodes that, so the SMT layer doesn't need to be taught
about aggregates as a new construct.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    AggregateVariable,
    ColumnVariable,
    DRCExpression,
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
    """Apply the aggregate operator (promote a column to an aggregate).

    See the module docstring for semantics. Returns an
    :class:`OperatorFailure` when validation rejects the application
    so the planner can retry with different inputs / params.
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

    if len(relation.result_variables) != 1:
        return OperatorFailure(
            error=(
                "Aggregate requires the input relation to have exactly one "
                f"result variable, got {len(relation.result_variables)}. "
                "Project to a single column first, then aggregate."
            )
        )

    rv = relation.result_variables[0]
    if not isinstance(rv, ColumnVariable):
        return OperatorFailure(
            error=(
                "Aggregate requires the input relation's single result "
                "variable to be a column (not already an aggregate). "
                f"Input has: {type(rv).__name__}"
            )
        )

    if rv.name != params.column:
        return OperatorFailure(
            error=(
                f"Aggregate column '{params.column}' does not match the "
                f"input relation's single column '{rv.name}'. The "
                "aggregate operator only promotes the existing result "
                "variable; project first if the column needs to change."
            )
        )

    output = DRCExpression(
        result_variables=[
            AggregateVariable(function=params.function, column=params.column)
        ],
        condition=relation.condition,
    )
    return OperatorSuccess(output=output)
