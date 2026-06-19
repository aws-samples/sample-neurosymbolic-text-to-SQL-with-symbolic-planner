"""Join (⋈) operator: natural join on specified columns.

Joins two relations on shared column names. The output contains the union
of columns from both inputs, with join columns appearing only once.

The DRC semantics of R1 ⋈_b R2 where R1 has columns (a, b) and R2 has (b, c):
  {a, b, c | C1(a, b) ∧ C2(b, c)}

where C1 is the condition of R1 and C2 is the condition of R2, with the
join column 'b' shared between them (same variable name ensures equality).

When inputs have overlapping non-join column names, we rename them with
fresh variables and use existential quantifiers.
"""

from __future__ import annotations

from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    DRCCondition,
    LogicalConnectiveNode,
    ComparisonNode,
    QuantifierNode,
    VariableRefNode,
    MembershipNode,
    ResultVariable,
)
from text_to_sql_planner.types.operators import (
    OperatorResult,
    OperatorSuccess,
    OperatorFailure,
    JoinParams,
)


def _get_columns(expr: DRCExpression) -> list[str]:
    """Extract column names from a DRC expression's result variables."""
    return [rv.name if isinstance(rv, ColumnVariable) else getattr(rv, "column", f"__expr_{i}__") for i, rv in enumerate(expr.result_variables)]


def _rename_variable(condition: DRCCondition, old_name: str, new_name: str) -> DRCCondition:
    """Recursively rename a variable in a condition tree."""
    if isinstance(condition, VariableRefNode):
        if condition.name == old_name:
            return VariableRefNode(name=new_name)
        return condition
    elif isinstance(condition, MembershipNode):
        new_vars = [new_name if v == old_name else v for v in condition.variables]
        return MembershipNode(variables=new_vars, relation=condition.relation)
    elif isinstance(condition, LogicalConnectiveNode):
        return LogicalConnectiveNode(
            operator=condition.operator,
            left=_rename_variable(condition.left, old_name, new_name),
            right=_rename_variable(condition.right, old_name, new_name),
        )
    elif isinstance(condition, ComparisonNode):
        return ComparisonNode(
            operator=condition.operator,
            left=_rename_variable(condition.left, old_name, new_name),
            right=_rename_variable(condition.right, old_name, new_name),
        )
    elif isinstance(condition, QuantifierNode):
        new_vars = [new_name if v == old_name else v for v in condition.variables]
        return QuantifierNode(
            kind=condition.kind,
            variables=new_vars,
            body=_rename_variable(condition.body, old_name, new_name),
        )
    # For other node types, return as-is
    return condition


def apply_join(params: JoinParams, inputs: list[DRCExpression]) -> OperatorResult:
    """Apply join operator R ⋈_{join_columns} S.

    The join produces {output_cols | C1 ∧ C2} where the join columns
    use the same variable name in both conditions to enforce equality.

    Non-join columns that overlap between R1 and R2 are renamed with
    fresh variable names in R2's condition.
    """
    if len(inputs) != 2:
        return OperatorFailure(error="Join requires exactly 2 input relations")

    r1, r2 = inputs[0], inputs[1]
    join_columns = params.join_columns

    if not join_columns:
        return OperatorFailure(error="Join requires at least one join column")

    r1_columns = _get_columns(r1)
    r2_columns = _get_columns(r2)

    # Validate join columns exist in both inputs
    for col in join_columns:
        if col not in r1_columns:
            return OperatorFailure(
                error=f"Join column '{col}' not found in first input relation. "
                f"Available columns: {r1_columns}"
            )
        if col not in r2_columns:
            return OperatorFailure(
                error=f"Join column '{col}' not found in second input relation. "
                f"Available columns: {r2_columns}"
            )

    # Determine output columns: all from r1, then non-join columns from r2
    # If a non-join column in r2 has the same name as one in r1, rename it
    output_variables: list[ResultVariable] = list(r1.result_variables)
    r2_condition = r2.condition
    r2_renames: dict[str, str] = {}  # old_name -> new_name
    seen_names: set[str] = set()
    for rv in r1.result_variables:
        col_name = rv.name if isinstance(rv, ColumnVariable) else getattr(rv, "column", "__expr__")
        seen_names.add(col_name)

    for rv in r2.result_variables:
        col_name = rv.name if isinstance(rv, ColumnVariable) else getattr(rv, "column", "__expr__")
        if col_name in join_columns:
            continue  # Skip join columns (already in output from r1)
        if col_name in seen_names:
            # Name conflict — rename in r2
            new_name = f"{col_name}_r2"
            r2_renames[col_name] = new_name
            output_variables.append(ColumnVariable(name=new_name))
            seen_names.add(new_name)
        else:
            output_variables.append(rv)
            seen_names.add(col_name)

    # Apply renames to r2's condition
    for old_name, new_name in r2_renames.items():
        r2_condition = _rename_variable(r2_condition, old_name, new_name)

    # The output condition is simply: C1 ∧ C2
    # The join columns share the same variable name, which enforces equality
    output_condition = LogicalConnectiveNode(
        operator="and",
        left=r1.condition,
        right=r2_condition,
    )

    output = DRCExpression(
        result_variables=output_variables,
        condition=output_condition,
    )

    return OperatorSuccess(output=output)
