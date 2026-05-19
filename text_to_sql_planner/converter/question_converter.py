"""Question converter: transforms natural language questions into DRC expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from text_to_sql_planner.parser import parse, ParserSuccess, ParserFailure
from text_to_sql_planner.planner.llm_client import (
    LLMClientConfig,
    convert_question_to_drc,
)
from text_to_sql_planner.printer import pretty_print, PrintSuccess
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

    print(f"[question-converter] Converting question to DRC...")
    print(f"[question-converter] Question: {question}")
    print(f"[question-converter] Model: {config.model_id}")
    print(f"[question-converter] Max attempts: {max_attempts}")

    for attempt in range(1, max_attempts + 1):
        print(f"[question-converter] Attempt {attempt}/{max_attempts}...", flush=True)

        try:
            # Get DRC Lisp syntax from LLM
            if attempt == 1:
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
                raw_lisp = await convert_question_to_drc(
                    enhanced_question, schema, config
                )

            last_raw_output = raw_lisp
            print(f"[question-converter] LLM returned: {raw_lisp}")

            # Parse the Lisp syntax
            result = parse(raw_lisp)

            if isinstance(result, ParserSuccess):
                print(f"[question-converter] Parse succeeded!")
                pp = pretty_print(result.expression)
                if isinstance(pp, PrintSuccess):
                    print(f"[question-converter] DRC (pretty): {pp.output}")
                print(f"[question-converter] DRC (lisp):   {raw_lisp}")
                return ConversionSuccess(
                    expression=result.expression,
                    lisp_syntax=raw_lisp,
                )
            elif isinstance(result, ParserFailure):
                last_error = str(result.error)
                print(f"[question-converter] Parse failed: {last_error}")
            else:
                last_error = "Unknown parser result type"
                print(f"[question-converter] Parse failed: {last_error}")

        except (ImportError, RuntimeError) as e:
            last_error = str(e)
            last_raw_output = ""
            print(f"[question-converter] Error: {last_error}")
            # Don't retry on import/runtime errors - they won't resolve
            return ConversionError(
                message=f"LLM error: {last_error}",
                attempts=attempt,
                last_raw_output=last_raw_output,
            )
        except Exception as e:
            last_error = str(e)
            print(f"[question-converter] Exception: {last_error}")

    print(f"[question-converter] All {max_attempts} attempts failed.")
    return ConversionError(
        message=f"Failed to parse DRC after {max_attempts} attempts. Last error: {last_error}",
        attempts=max_attempts,
        last_raw_output=last_raw_output,
    )
