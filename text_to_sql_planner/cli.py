"""Command-line interface for the Text-to-SQL Planner."""

from __future__ import annotations

import argparse
import asyncio
import sys

from text_to_sql_planner.main import run, TextToSQLSuccess, TextToSQLFailure
from text_to_sql_planner.planner.planner import PlannerConfig
from text_to_sql_planner.planner.llm_client import LLMClientConfig
from text_to_sql_planner.equivalence import EquivalenceCheckerConfig
from text_to_sql_planner.printer import pretty_print, PrintSuccess


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

    # Run the pipeline
    result = asyncio.run(_run_pipeline(question, schema, config, verbose=args.verbose))
    sys.exit(result)


async def _run_pipeline(
    question: str, schema: str, config: PlannerConfig, verbose: bool
) -> int:
    """Run the text-to-SQL pipeline and print results."""
    result = await run(question=question, schema=schema, config=config)

    if isinstance(result, TextToSQLSuccess):
        if verbose:
            # Print the target DRC expression
            pp_result = pretty_print(result.target_expression)
            if isinstance(pp_result, PrintSuccess):
                print(f"Target DRC: {pp_result.output}", file=sys.stderr)
            print(f"Iterations: {result.operation_tree}", file=sys.stderr)
            print("---", file=sys.stderr)

        print(result.sql)
        return 0

    else:
        print(f"Error [{result.code.value}]: {result.error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
