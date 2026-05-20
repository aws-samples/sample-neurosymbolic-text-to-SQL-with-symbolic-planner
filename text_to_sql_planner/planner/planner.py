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
    select_operator,
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
    DivisionParams,
    JoinParams,
    OperatorApplication,
    OperatorSuccess,
    OperatorFailure,
    ProjectionParams,
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
    elif operator == "division":
        return DivisionParams()
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

    # Prepare table relation descriptions for the LLM
    table_descs = []
    for tr in table_relations:
        tr_print = print_lisp(tr.expression)
        lisp_str = tr_print.output if isinstance(tr_print, PrintSuccess) else ""
        table_descs.append(
            {
                "table_name": tr.table_name,
                "columns": tr.columns,
                "lisp_syntax": lisp_str,
            }
        )
        print(f"  - `{tr.table_name}` columns=`{tr.columns}`")

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

    for iteration in range(1, config.max_iterations + 1):
        print(f"\n## Iteration {iteration}/{config.max_iterations}\n", flush=True)
        print(f"### Available relations ({len(available)})\n")
        for i, (expr, cols, node) in enumerate(available):
            if isinstance(node, TableLeafNode):
                label = f"Table `{node.table_name}`"
            else:
                label = f"Intermediate (`{node.operator}`)"
            pp = pretty_print_indented(expr)
            drc_str = pp.output if isinstance(pp, PrintSuccess) else f"columns={cols}"
            print(f"**[{i}]** {label} — columns=`{cols}`\n")
            print(f"```\n{drc_str}\n```\n")
        success_this_iteration = False

        for retry in range(config.max_retries_per_iteration):
            temperature = config.initial_temperature + (retry * config.temperature_step)
            temperature = min(temperature, 1.0)

            if retry > 0:
                print(f"> ⚠️ Retry {retry}/{config.max_retries_per_iteration} (temp={temperature:.1f})\n")

            try:
                print(f"Asking LLM to select operator (temp={temperature:.1f})...\n", flush=True)
                selection = await select_operator(
                    table_relations=table_descs,
                    intermediate_relations=intermediate_relations,
                    target_relation=target_lisp,
                    temperature=temperature,
                    config=config.llm_config,
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
                print(f"> ⚠️ Invalid: no input indices provided\n")
                continue
            if any(idx < 0 or idx >= len(available) for idx in selection.input_indices):
                print(f"> ⚠️ Invalid: input indices out of range (have {len(available)} available)\n")
                continue

            # Build operator params
            op_params = _build_operator_params(selection.operator, selection.params)
            if op_params is None:
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
                print(f"> ❌ Operator failed: `{op_result.error}`\n")
                continue

            # We have a new intermediate expression
            new_expr = op_result.output
            new_columns = _get_columns_from_expression(new_expr)
            print(f"✅ **Operator succeeded** — output columns: `{new_columns}`\n")

            new_pp = pretty_print_indented(new_expr)
            new_lisp_result = print_lisp(new_expr)
            new_lisp_str = new_lisp_result.output if isinstance(new_lisp_result, PrintSuccess) else ""
            if isinstance(new_pp, PrintSuccess):
                print(f"```\n{new_pp.output}\n```\n")

            # Check for duplicates: skip if this expression already exists
            is_duplicate = False
            for existing_expr, existing_cols, _ in available:
                existing_lisp = print_lisp(existing_expr)
                if isinstance(existing_lisp, PrintSuccess) and existing_lisp.output == new_lisp_str:
                    is_duplicate = True
                    break
            if is_duplicate:
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
            intermediate_relations.append(
                IntermediateRelation(
                    index=new_index,
                    expression=new_expr,
                    columns=new_columns,
                )
            )
            print(f"Added as **relation [{new_index}]**\n")

            # Check equivalence with target
            print(f"Checking equivalence with target...\n", flush=True)
            eq_result = await check_equivalence(
                new_expr, target_relation, config.equivalence_config
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
