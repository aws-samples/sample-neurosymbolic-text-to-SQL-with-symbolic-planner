"""Planner loop: iteratively builds an operation tree using LLM-guided operator selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from text_to_sql_planner.converter.table_converter import TableRelation
from text_to_sql_planner.equivalence import (
    EquivalenceCheckerConfig,
    EquivalentResult,
    NotEquivalentResult,
    IndeterminateResult,
    check_equivalence,
)
from text_to_sql_planner.operators import apply_operator
from text_to_sql_planner.parser import parse, ParserSuccess
from text_to_sql_planner.planner.llm_client import (
    IntermediateRelation,
    LLMClientConfig,
    OperatorSelection,
    RejectedProposal,
    select_operator,
    summarize_relation,
)
from text_to_sql_planner.printer import print_lisp, pretty_print, pretty_print_indented, PrintSuccess
from text_to_sql_planner.types.drc import DRCExpression
from text_to_sql_planner.types.operation_tree import (
    OperationTree,
    OperatorNode,
    TableLeafNode,
)
from text_to_sql_planner.types.operators import (
    CartesianProductParams,
    DifferenceParams,
    DivisionParams,
    JoinParams,
    OperatorApplication,
    OperatorSuccess,
    OperatorFailure,
    ProjectionParams,
    RenameParams,
    SelectionParams,
    UnionParams,
)


@dataclass
class PlannerConfig:
    """Configuration for the planner loop."""

    max_iterations: int = 50
    max_retries_per_iteration: int = 5
    initial_temperature: float = 0.0
    temperature_step: float = 0.2
    llm_config: LLMClientConfig = field(default_factory=LLMClientConfig)
    equivalence_config: EquivalenceCheckerConfig = field(
        default_factory=EquivalenceCheckerConfig
    )


@dataclass
class PlannerSuccess:
    """Successful planning result."""

    operation_tree: OperationTree
    iterations: int


@dataclass
class PlannerError:
    """Failed planning result."""

    error_type: str  # 'max_iterations_exceeded' | 'operator_selection_failed'
    message: str
    iterations: int


PlannerResult = Union[PlannerSuccess, PlannerError]


def _get_columns_from_expression(expr: DRCExpression) -> list[str]:
    """Extract column names from a DRC expression's result variables."""
    columns = []
    for rv in expr.result_variables:
        if hasattr(rv, "name"):
            columns.append(rv.name)
        elif hasattr(rv, "column"):
            columns.append(rv.column)
    return columns


def _build_operator_params(operator: str, params: dict) -> object:
    """Convert raw params dict to the appropriate OperatorParams dataclass."""
    if operator == "selection":
        condition = params.get("condition")
        if isinstance(condition, str):
            wrapped = f"(drc (x) {condition})"
            result = parse(wrapped)
            if isinstance(result, ParserSuccess):
                condition = result.expression.condition
            else:
                return None
        return SelectionParams(condition=condition)
    elif operator == "projection":
        return ProjectionParams(columns=params.get("columns", []))
    elif operator == "join":
        return JoinParams(join_columns=params.get("join_columns", []))
    elif operator == "cartesian_product":
        return CartesianProductParams()
    elif operator == "union":
        return UnionParams()
    elif operator == "difference":
        return DifferenceParams()
    elif operator == "division":
        return DivisionParams()
    elif operator == "rename":
        # ``mapping`` arrives as a dict ``{old_name: new_name, ...}``.
        # Validation is delegated to ``apply_rename``; this just
        # builds the params dataclass.
        raw_mapping = params.get("mapping", {})
        if not isinstance(raw_mapping, dict):
            return None
        return RenameParams(mapping=dict(raw_mapping))
    return None


async def plan(
    table_relations: list[TableRelation],
    target_relation: DRCExpression,
    config: PlannerConfig | None = None,
) -> PlannerResult:
    """Run the planner loop to build an operation tree."""
    if config is None:
        config = PlannerConfig()

    # Convert target to lisp syntax for the LLM
    target_print = print_lisp(target_relation)
    if not isinstance(target_print, PrintSuccess):
        return PlannerError(
            error_type="operator_selection_failed",
            message="Failed to print target expression to Lisp syntax.",
            iterations=0,
        )
    target_lisp = target_print.output

    target_pp = pretty_print_indented(target_relation)
    target_pretty = target_pp.output if isinstance(target_pp, PrintSuccess) else target_lisp

    print(f"# Planner\n")
    print(f"**Target (pretty):**\n```\n{target_pretty}\n```\n")
    print(f"**Target (lisp):**\n```lisp\n{target_lisp}\n```\n")
    print(f"- **Max iterations:** {config.max_iterations}")
    print(f"- **Max retries/iteration:** {config.max_retries_per_iteration}")
    print(f"- **Tables:** {[tr.table_name for tr in table_relations]}\n")

    # Build schema types from table relations (for SMT-LIB type inference)
    schema_types: dict[str, str] = {}
    for tr in table_relations:
        if hasattr(tr, 'column_types'):
            for col, col_type in tr.column_types.items():
                if col_type == "String":
                    schema_types[col] = "String"

    # Prepare table relation descriptions for the LLM
    table_descs = []
    # Track summaries for each available relation
    summaries: list[str] = []
    for tr in table_relations:
        tr_print = print_lisp(tr.expression)
        lisp_str = tr_print.output if isinstance(tr_print, PrintSuccess) else ""
        summary = f"All rows from table {tr.table_name}"
        table_descs.append(
            {
                "table_name": tr.table_name,
                "columns": tr.columns,
                "lisp_syntax": lisp_str,
                "summary": summary,
            }
        )
        summaries.append(summary)
        print(f"  - `{tr.table_name}` columns=`{tr.columns}` — _{summary}_")

    # Track available expressions (tables + intermediates)
    available: list[tuple[DRCExpression, list[str], object]] = []
    for tr in table_relations:
        leaf = TableLeafNode(
            table_name=tr.table_name,
            columns=tr.columns,
            expression=tr.expression,
        )
        available.append((tr.expression, tr.columns, leaf))

    intermediate_relations: list[IntermediateRelation] = []

    # Degenerate case: check if the target is already one of the input tables
    target_columns = _get_columns_from_expression(target_relation)
    target_lisp_for_compare = print_lisp(target_relation)
    target_lisp_str = target_lisp_for_compare.output if isinstance(target_lisp_for_compare, PrintSuccess) else ""

    for i, (expr, cols, node) in enumerate(available):
        if cols == target_columns:
            expr_lisp = print_lisp(expr)
            if isinstance(expr_lisp, PrintSuccess) and expr_lisp.output == target_lisp_str:
                print(f"### ✅ Degenerate case: target is already table relation [{i}]\n")
                tree = OperationTree(root=node)
                return PlannerSuccess(operation_tree=tree, iterations=0)

    # Also check via cvc5 equivalence for structural matches
    for i, (expr, cols, node) in enumerate(available):
        if len(cols) == len(target_columns):
            eq_result = await check_equivalence(
                expr, target_relation, config.equivalence_config,
                schema_types=schema_types,
                label=f"planner: candidate relation [{i}] vs target DRC",
                lhs_label=f"candidate [{i}]",
                rhs_label="target",
            )
            if isinstance(eq_result, EquivalentResult):
                print(f"### ✅ Degenerate case: target is equivalent to table relation [{i}]\n")
                tree = OperationTree(root=node)
                return PlannerSuccess(operation_tree=tree, iterations=0)

    for iteration in range(1, config.max_iterations + 1):
        print(f"\n## Iteration {iteration}/{config.max_iterations}\n", flush=True)
        print(f"### Available relations ({len(available)})\n")
        for i, (expr, cols, node) in enumerate(available):
            if isinstance(node, TableLeafNode):
                label = f"Table `{node.table_name}`"
            else:
                label = f"Intermediate (`{node.operator}`)"
            summary = summaries[i] if i < len(summaries) else ""
            pp = pretty_print_indented(expr)
            drc_str = pp.output if isinstance(pp, PrintSuccess) else f"columns={cols}"
            print(f"**[{i}]** {label} — _{summary}_\n")
            print(f"```\n{drc_str}\n```\n")
        success_this_iteration = False

        # Negative memory for this iteration: every retry that fails for any
        # reason (bad params, duplicate output, operator failure, not-
        # equivalent verdict, ...) gets recorded here and surfaced back to
        # the LLM on the next retry so it doesn't loop on the same proposal.
        # Reset between iterations because once a proposal succeeds we have
        # a new starting point and old rejections may no longer apply.
        rejected_proposals: list[RejectedProposal] = []

        def _record_rejection(sel: OperatorSelection | None, reason: str) -> None:
            """Append a rejection to the iteration's negative-memory list.

            ``sel`` may be ``None`` when the LLM call itself failed before
            returning a parseable selection — there's nothing to forbid in
            that case so we only log.
            """
            if sel is None:
                return
            rejected_proposals.append(
                RejectedProposal(
                    operator=sel.operator,
                    input_indices=list(sel.input_indices),
                    params=dict(sel.params),
                    reason=reason,
                )
            )

        for retry in range(config.max_retries_per_iteration):
            temperature = config.initial_temperature + (retry * config.temperature_step)
            temperature = min(temperature, 1.0)

            if retry > 0:
                print(f"> ⚠️ Retry {retry}/{config.max_retries_per_iteration} (temp={temperature:.1f})\n")

            selection: OperatorSelection | None = None
            try:
                print(f"Asking LLM to select operator (temp={temperature:.1f})...\n", flush=True)
                selection = await select_operator(
                    table_relations=table_descs,
                    intermediate_relations=intermediate_relations,
                    target_relation=target_lisp,
                    temperature=temperature,
                    config=config.llm_config,
                    rejected_proposals=rejected_proposals,
                )
                print(f"**LLM selected:** `{selection.operator}` inputs=`{selection.input_indices}` params=`{selection.params}`\n")
                # For joins, show a clearer breakdown
                if selection.operator == "join" and selection.params.get("join_columns"):
                    join_cols = selection.params["join_columns"]
                    input_cols = [available[idx][1] if 0 <= idx < len(available) else [] for idx in selection.input_indices]
                    left_cols = input_cols[0] if len(input_cols) > 0 else []
                    right_cols = input_cols[1] if len(input_cols) > 1 else []
                    left_other = [c for c in left_cols if c not in join_cols]
                    right_other = [c for c in right_cols if c not in join_cols]
                    print(f"| | Join columns | Other columns |")
                    print(f"|---|---|---|")
                    print(f"| Left [{selection.input_indices[0]}] | `{join_cols}` | `{left_other}` |")
                    print(f"| Right [{selection.input_indices[1]}] | `{join_cols}` | `{right_other}` |")
                    print(f"| **Output** | | `{join_cols + left_other + right_other}` |")
                    print()
                if selection.reasoning:
                    print(f"#### Reasoning\n\n{selection.reasoning}\n")
            except Exception as e:
                print(f"> ❌ LLM call failed: `{e}`\n")
                continue

            # Validate input indices
            if not selection.input_indices:
                _record_rejection(selection, "no input indices provided")
                print(f"> ⚠️ Invalid: no input indices provided\n")
                continue
            if any(idx < 0 or idx >= len(available) for idx in selection.input_indices):
                _record_rejection(
                    selection,
                    f"input indices out of range (have {len(available)} available)",
                )
                print(f"> ⚠️ Invalid: input indices out of range (have {len(available)} available)\n")
                continue

            # Build operator params
            op_params = _build_operator_params(selection.operator, selection.params)
            if op_params is None:
                _record_rejection(selection, "could not build operator params")
                print(f"> ⚠️ Invalid: could not build operator params\n")
                continue

            # Gather input expressions
            input_exprs = [available[idx][0] for idx in selection.input_indices]

            # Apply the operator
            application = OperatorApplication(
                operator=selection.operator,
                inputs=input_exprs,
                params=op_params,
            )
            op_result = apply_operator(application)

            if isinstance(op_result, OperatorFailure):
                _record_rejection(selection, f"operator failed: {op_result.error}")
                print(f"> ❌ Operator failed: `{op_result.error}`\n")
                continue

            # We have a new intermediate expression
            new_expr = op_result.output
            # Simplify the DRC (merge nested quantifiers, etc.)
            from text_to_sql_planner.drc_simplifier import simplify_drc
            new_expr = simplify_drc(new_expr)
            new_columns = _get_columns_from_expression(new_expr)
            print(f"✅ **Operator succeeded** — output columns: `{new_columns}`\n")

            new_pp = pretty_print_indented(new_expr)
            new_lisp_result = print_lisp(new_expr)
            new_lisp_str = new_lisp_result.output if isinstance(new_lisp_result, PrintSuccess) else ""
            if isinstance(new_pp, PrintSuccess):
                print(f"```\n{new_pp.output}\n```\n")

            # Check for duplicates: skip if this expression already exists
            duplicate_index: int | None = None
            for existing_idx, (existing_expr, existing_cols, _) in enumerate(available):
                existing_lisp = print_lisp(existing_expr)
                if isinstance(existing_lisp, PrintSuccess) and existing_lisp.output == new_lisp_str:
                    duplicate_index = existing_idx
                    break
            if duplicate_index is not None:
                _record_rejection(
                    selection,
                    f"output is identical to existing relation [{duplicate_index}]",
                )
                print(f"> ⚠️ **DUPLICATE** — skipping (already have this relation)\n")
                continue

            # Build the tree node for this operation
            input_nodes = [available[idx][2] for idx in selection.input_indices]
            op_node = OperatorNode(
                operator=selection.operator,
                params=op_params,
                inputs=input_nodes,
                output_expression=new_expr,
                output_columns=new_columns,
            )

            # Add to available relations
            new_index = len(available)
            available.append((new_expr, new_columns, op_node))

            # Summarize the new relation
            new_pp_for_summary = pretty_print(new_expr)
            drc_for_summary = new_pp_for_summary.output if isinstance(new_pp_for_summary, PrintSuccess) else ""
            try:
                summary = await summarize_relation(drc_for_summary, new_columns, config.llm_config)
            except Exception:
                summary = f"Result of {selection.operator} on [{', '.join(str(i) for i in selection.input_indices)}]"
            summaries.append(summary)

            intermediate_relations.append(
                IntermediateRelation(
                    index=new_index,
                    expression=new_expr,
                    columns=new_columns,
                    summary=summary,
                )
            )
            print(f"Added as **relation [{new_index}]** — _{summary}_\n")

            # Check equivalence with target
            print(f"Checking equivalence with target...\n", flush=True)
            eq_result = await check_equivalence(
                new_expr, target_relation, config.equivalence_config,
                schema_types=schema_types,
                label=(
                    f"planner: just-built relation [{new_index}] "
                    f"vs target DRC (iteration {iteration})"
                ),
                lhs_label=f"built [{new_index}]",
                rhs_label="target",
            )

            if isinstance(eq_result, EquivalentResult):
                print(f"### ✅ EQUIVALENT — Planning complete!\n")
                tree = OperationTree(root=op_node)
                return PlannerSuccess(
                    operation_tree=tree,
                    iterations=iteration,
                )
            elif isinstance(eq_result, NotEquivalentResult):
                print(f"> Not equivalent yet, continuing...\n")
            elif isinstance(eq_result, IndeterminateResult):
                print(f"> ⚠️ Equivalence indeterminate: {eq_result.reason} — treating as not-equivalent\n")

            success_this_iteration = True
            break

        if not success_this_iteration:
            print(f"\n> ❌ All retries exhausted at iteration {iteration}. Stopping.\n")
            return PlannerError(
                error_type="operator_selection_failed",
                message=(
                    f"Failed to select a valid operator after "
                    f"{config.max_retries_per_iteration} retries at iteration {iteration}."
                ),
                iterations=iteration,
            )

    print(f"\n> ❌ Max iterations ({config.max_iterations}) reached without equivalence.\n")
    return PlannerError(
        error_type="max_iterations_exceeded",
        message=f"Reached maximum iterations ({config.max_iterations}) without finding equivalence.",
        iterations=config.max_iterations,
    )
