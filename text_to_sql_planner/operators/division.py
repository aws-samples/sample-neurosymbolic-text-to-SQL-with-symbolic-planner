"""Division (÷) operator: finds tuples in R associated with all tuples in S.

Division R ÷ S returns tuples from R (projected onto columns not in S)
that are associated with every tuple in S. The second relation's columns
must be a subset of the first relation's columns.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    QuantifierNode,
    LogicalConnectiveNode,
    MembershipNode,
    ResultVariable,
)
from text_to_sql_planner.types.operators import (
    OperatorResult,
    OperatorSuccess,
    OperatorFailure,
    DivisionParams,
)


def _get_columns(expr: DRCExpression) -> list[str]:
    """Extract column names from a DRC expression's result variables."""
    return [rv.name if isinstance(rv, ColumnVariable) else rv.column for rv in expr.result_variables]


def apply_division(params: DivisionParams, inputs: list[DRCExpression]) -> OperatorResult:
    """Apply division operator R ÷ S.

    Takes two input relations. Validates that the second relation's columns
    are a subset of the first relation's columns. Output columns are the
    first input's columns minus the second input's columns. Output condition
    uses a forall quantifier over the second relation.
    """
    if len(inputs) != 2:
        return OperatorFailure(error="Division requires exactly 2 input relations")

    r1, r2 = inputs[0], inputs[1]

    r1_columns = _get_columns(r1)
    r2_columns = _get_columns(r2)

    # Validate r2 columns are a subset of r1 columns
    for col in r2_columns:
        if col not in r1_columns:
            return OperatorFailure(
                error=f"Division requires second relation's columns to be a subset "
                f"of first relation's columns. Column '{col}' from second relation "
                f"not found in first relation. First relation columns: {r1_columns}"
            )

    # Output columns: r1 columns minus r2 columns
    output_col_names = [col for col in r1_columns if col not in r2_columns]
    output_variables: list[ResultVariable] = []
    for rv in r1.result_variables:
        col_name = rv.name if isinstance(rv, ColumnVariable) else rv.column
        if col_name in output_col_names:
            output_variables.append(rv)

    if not output_variables:
        return OperatorFailure(
            error="Division would produce an empty set of output columns. "
            "Second relation's columns cannot be identical to first relation's columns."
        )

    # Output condition: forall quantifier over r2's columns
    # (forall (r2_columns) (and (in r2_columns r2) r1_condition))
    # This means: for all values of r2's columns, if they're in R2 then they're in R1
    output_condition = QuantifierNode(
        kind="forall",
        variables=r2_columns,
        body=LogicalConnectiveNode(
            operator="and",
            left=r2.condition,
            right=r1.condition,
        ),
    )

    output = DRCExpression(
        result_variables=output_variables,
        condition=output_condition,
    )

    return OperatorSuccess(output=output)


def _extract_relation_name(expr: DRCExpression) -> str:
    """Extract a relation name from a DRC expression's condition."""
    condition = expr.condition
    if isinstance(condition, MembershipNode):
        return condition.relation
    return _find_relation_name(condition)


def _find_relation_name(node) -> str:
    """Recursively search for a relation name in a condition tree."""
    if node is None:
        return "R"
    if isinstance(node, MembershipNode):
        return node.relation
    if isinstance(node, QuantifierNode):
        return _find_relation_name(node.body)
    if isinstance(node, LogicalConnectiveNode):
        result = _find_relation_name(node.left)
        if result != "R":
            return result
        return _find_relation_name(node.right)
    return "R"
