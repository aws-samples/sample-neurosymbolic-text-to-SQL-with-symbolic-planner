"""Command-line interface for the Text-to-SQL Planner."""

from __future__ import annotations

import argparse
import asyncio
import sys

from text_to_sql_planner.main import run, TextToSQLSuccess, TextToSQLFailure
from text_to_sql_planner.planner.planner import PlannerConfig
from text_to_sql_planner.planner.llm_client import LLMClientConfig
from text_to_sql_planner.equivalence import EquivalenceCheckerConfig
from text_to_sql_planner.printer import pretty_print, pretty_print_query, PrintSuccess


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="text-to-sql",
        description="Convert natural language questions into SQL queries using a symbolic planner.",
    )

    # Input options (mutually exclusive: inline vs file)
    input_group = parser.add_argument_group("input")
    input_group.add_argument(
        "-q", "--question",
        help="The natural language question to convert to SQL.",
    )
    input_group.add_argument(
        "-s", "--schema",
        help="The database schema as a string (CREATE TABLE statements).",
    )
    input_group.add_argument(
        "-f", "--schema-file",
        help="Path to a file containing the database schema.",
    )

    # Configuration options
    config_group = parser.add_argument_group("configuration")
    config_group.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1).",
    )
    config_group.add_argument(
        "--model-id",
        default="global.anthropic.claude-opus-4-6-v1",
        help="Bedrock model ID (default: global.anthropic.claude-opus-4-6-v1).",
    )
    config_group.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum planner iterations (default: 50).",
    )
    config_group.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per iteration (default: 5).",
    )
    config_group.add_argument(
        "--cvc5-path",
        default="cvc5",
        help="Path to the cvc5 binary (default: cvc5).",
    )
    config_group.add_argument(
        "--cvc5-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for cvc5 equivalence checks (default: 30).",
    )

    # Output options
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show the target DRC expression and operation tree details.",
    )
    output_group.add_argument(
        "-o", "--output",
        help="Write Markdown output to a file (default: stdout).",
    )

    args = parser.parse_args()

    # Resolve question
    question = args.question
    if not question:
        # Read from stdin if no -q flag
        if not sys.stdin.isatty():
            question = sys.stdin.read().strip()
        else:
            parser.error("Provide a question with -q or pipe it via stdin.")

    # Resolve schema
    schema = args.schema
    if args.schema_file:
        try:
            with open(args.schema_file, "r") as f:
                schema = f.read()
        except (OSError, IOError) as e:
            print(f"Error reading schema file: {e}", file=sys.stderr)
            sys.exit(1)

    if not schema:
        parser.error("Provide a schema with -s or --schema-file.")

    # Build config
    config = PlannerConfig(
        max_iterations=args.max_iterations,
        max_retries_per_iteration=args.max_retries,
        llm_config=LLMClientConfig(
            region=args.region,
            model_id=args.model_id,
        ),
        equivalence_config=EquivalenceCheckerConfig(
            cvc5_path=args.cvc5_path,
            timeout_seconds=args.cvc5_timeout,
        ),
    )

    # Redirect stdout to file if -o specified
    output_file = None
    if args.output:
        try:
            output_file = open(args.output, "w")
            sys.stdout = output_file
        except (OSError, IOError) as e:
            print(f"Error opening output file: {e}", file=sys.stderr)
            sys.exit(1)

    # Run the pipeline
    try:
        result = asyncio.run(_run_pipeline(question, schema, config, verbose=args.verbose))
    finally:
        if output_file:
            sys.stdout = sys.__stdout__
            output_file.close()
            print(f"Output written to {args.output}", file=sys.stderr)

    sys.exit(result)


async def _run_pipeline(
    question: str, schema: str, config: PlannerConfig, verbose: bool
) -> int:
    """Run the text-to-SQL pipeline and print results."""
    from text_to_sql_planner.types.operation_tree import TableLeafNode, OperatorNode

    result = await run(question=question, schema=schema, config=config)

    if isinstance(result, TextToSQLSuccess):
        print(f"\n---\n")
        print(f"# Result\n")
        if verbose:
            pp_result = pretty_print_query(result.target_query)
            if isinstance(pp_result, PrintSuccess):
                print(f"**Target Query:**\n```\n{pp_result.output}\n```\n")

        print(f"**Generated SQL:**\n")
        print(f"```sql\n{result.sql}\n```\n")

        # Simplify
        from text_to_sql_planner.sql import simplify_sql
        simplified = simplify_sql(result.sql)
        if simplified != result.sql:
            print(f"**Simplified SQL:**\n")
            print(f"```sql\n{simplified}\n```\n")

        # Print the operation tree
        print(f"## Operation Tree\n")
        print(f"```")
        _print_tree(result.operation_tree.root, indent=0)
        print(f"```\n")

        sys.stdout.flush()
        return 0

    else:
        print(f"\n---\n")
        print(f"# ❌ Error\n")
        print(f"- **Code:** `{result.code.value}`")
        print(f"- **Message:** {result.error}\n")
        sys.stdout.flush()
        return 1


def _print_tree(node, indent: int = 0) -> None:
    """Print an operation tree node recursively with indentation."""
    from text_to_sql_planner.types.operation_tree import TableLeafNode, OperatorNode
    from text_to_sql_planner.types.operators import (
        SelectionParams, JoinParams, ProjectionParams,
    )

    pad = "  " * indent
    if isinstance(node, TableLeafNode):
        print(f"{pad}📋 Table: {node.table_name} [{', '.join(node.columns)}]")
    elif isinstance(node, OperatorNode):
        params_str = ""
        if isinstance(node.params, SelectionParams) and node.params.condition:
            from text_to_sql_planner.printer import pretty_print as pp
            from text_to_sql_planner.types.drc import DRCExpression, ColumnVariable
            # Print the condition
            cond_expr = DRCExpression(
                result_variables=[ColumnVariable(name="x")],
                condition=node.params.condition,
            )
            pp_result = pp(cond_expr)
            if isinstance(pp_result, PrintSuccess):
                # Extract just the condition part (after "| ")
                cond_str = pp_result.output.split("| ", 1)[-1].rstrip("}")
                params_str = f" WHERE {cond_str}"
        elif isinstance(node.params, JoinParams):
            params_str = f" ON [{', '.join(node.params.join_columns)}]"
        elif isinstance(node.params, ProjectionParams):
            params_str = f" [{', '.join(node.params.columns)}]"

        print(f"{pad}🔧 {node.operator}{params_str} → [{', '.join(node.output_columns)}]")
        for child in node.inputs:
            _print_tree(child, indent + 1)
    else:
        print(f"{pad}? Unknown node: {type(node).__name__}")


if __name__ == "__main__":
    main()
