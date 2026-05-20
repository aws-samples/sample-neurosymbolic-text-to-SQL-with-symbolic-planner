"""Question converter: transforms natural language questions into DRC expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from text_to_sql_planner.parser import parse, ParserSuccess, ParserFailure
from text_to_sql_planner.planner.llm_client import (
    LLMClientConfig,
    convert_question_to_drc,
)
from text_to_sql_planner.printer import pretty_print, pretty_print_indented, PrintSuccess
from text_to_sql_planner.types.drc import DRCExpression


@dataclass
class ConversionSuccess:
    """Successful question-to-DRC conversion."""

    expression: DRCExpression
    lisp_syntax: str  # raw LLM output


@dataclass
class ConversionError:
    """Failed question-to-DRC conversion."""

    message: str
    attempts: int
    last_raw_output: str = ""


ConversionResult = Union[ConversionSuccess, ConversionError]


async def convert_question(
    question: str,
    schema: str,
    config: LLMClientConfig | None = None,
    max_attempts: int = 3,
) -> ConversionResult:
    """Convert a natural language question into a DRC expression.

    Uses the LLM to generate DRC Lisp syntax, then parses it.
    On parse failure, retries with error feedback up to max_attempts.

    Args:
        question: The natural language question.
        schema: The database schema (CREATE TABLE statements).
        config: LLM client configuration.
        max_attempts: Maximum number of attempts (default 3).

    Returns:
        ConversionSuccess with the parsed expression, or
        ConversionError with details about the failure.
    """
    if config is None:
        config = LLMClientConfig()

    last_raw_output = ""
    last_error = ""

    print(f"## Question → DRC Conversion\n")
    print(f"- **Question:** {question}")
    print(f"- **Model:** {config.model_id}")
    print(f"- **Max attempts:** {max_attempts}\n")

    for attempt in range(1, max_attempts + 1):
        print(f"### Attempt {attempt}/{max_attempts}\n", flush=True)

        try:
            # Get DRC Lisp syntax from LLM
            if attempt == 1:
                user_msg = f"Schema:\n{schema}\n\nQuestion: {question}"
                print(f"##### System prompt\n")
                from text_to_sql_planner.planner.llm_client import _SYSTEM_PROMPT_DRC
                print(f"```\n{_SYSTEM_PROMPT_DRC.strip()}\n```\n")
                print(f"##### User message\n")
                print(f"```\n{user_msg}\n```\n")
                raw_lisp = await convert_question_to_drc(question, schema, config)
            else:
                # Include error feedback in subsequent attempts
                enhanced_question = (
                    f"{question}\n\n"
                    f"[Previous attempt produced invalid syntax. "
                    f"Error: {last_error}. "
                    f"Previous output: {last_raw_output}. "
                    f"Please fix the syntax.]"
                )
                user_msg = f"Schema:\n{schema}\n\nQuestion: {enhanced_question}"
                print(f"#### LLM prompt (with error feedback)\n")
                print(f"```\n{user_msg}\n```\n")
                raw_lisp = await convert_question_to_drc(
                    enhanced_question, schema, config
                )

            last_raw_output = raw_lisp
            print(f"#### LLM raw output\n")
            print(f"```lisp\n{raw_lisp}\n```\n")

            # Parse the Lisp syntax
            result = parse(raw_lisp)

            if isinstance(result, ParserSuccess):
                print(f"✅ **Parse succeeded**\n")
                pp = pretty_print_indented(result.expression)
                if isinstance(pp, PrintSuccess):
                    print(f"**DRC (pretty):**\n```\n{pp.output}\n```\n")
                print(f"**DRC (lisp):**\n```lisp\n{raw_lisp}\n```\n")
                return ConversionSuccess(
                    expression=result.expression,
                    lisp_syntax=raw_lisp,
                )
            elif isinstance(result, ParserFailure):
                last_error = str(result.error)
                print(f"❌ **Parse failed:** {last_error}\n")
            else:
                last_error = "Unknown parser result type"
                print(f"❌ **Parse failed:** {last_error}\n")

        except (ImportError, RuntimeError) as e:
            last_error = str(e)
            last_raw_output = ""
            print(f"❌ **Error:** {last_error}\n")
            # Don't retry on import/runtime errors - they won't resolve
            return ConversionError(
                message=f"LLM error: {last_error}",
                attempts=attempt,
                last_raw_output=last_raw_output,
            )
        except Exception as e:
            last_error = str(e)
            print(f"[question-converter] Exception: {last_error}")

    print(f"\n## All {max_attempts} attempts failed.\n")
    return ConversionError(
        message=f"Failed to parse DRC after {max_attempts} attempts. Last error: {last_error}",
        attempts=max_attempts,
        last_raw_output=last_raw_output,
    )
