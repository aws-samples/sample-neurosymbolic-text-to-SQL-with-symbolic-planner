"""Question converter: transforms natural language questions into DRC expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from text_to_sql_planner.parser import (
    parse_query,
    QueryParserSuccess,
    QueryParserFailure,
)
from text_to_sql_planner.planner.llm_client import (
    LLMClientConfig,
    convert_question_to_drc,
    decide_distinct,
)
from text_to_sql_planner.printer import (
    pretty_print,
    pretty_print_indented,
    pretty_print_query,
    PrintSuccess,
)
from text_to_sql_planner.types.drc import (
    DRCExpression,
    QueryExpression,
    query_inner_drc,
)


@dataclass
class ConversionSuccess:
    """Successful question-to-DRC conversion."""

    expression: DRCExpression  # the inner core DRC, for backward compatibility
    lisp_syntax: str  # raw LLM output
    query: QueryExpression | None = None  # the full extended-DRC query (None defaults to ``expression``)
    use_distinct: bool = False  # whether the question implies SELECT DISTINCT
    distinct_reasoning: str = ""

    def __post_init__(self) -> None:
        # If only the inner DRC was supplied (e.g. by tests), treat it as
        # an unwrapped query.
        if self.query is None:
            self.query = self.expression


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
                # Retry with error feedback only — DO NOT include the prior
                # malformed output. Models tend to anchor on the text they
                # see and reproduce the same bug, so we describe what went
                # wrong and ask for a fresh attempt.
                enhanced_question = (
                    f"{question}\n\n"
                    f"[The previous attempt produced syntactically invalid "
                    f"DRC Lisp. Parser error: {last_error}. "
                    f"Common causes are unbalanced parentheses (more "
                    f"closing than opening, or vice versa) and over-deep "
                    f"nesting in the negation of an \"exactly N\" pattern. "
                    f"Produce a fresh, well-formed expression, counting "
                    f"the parentheses carefully.]"
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

            # Parse the Lisp syntax (extended DRC: optional limit / order-by
            # wrappers around a core ``(drc ...)`` form).
            result = parse_query(raw_lisp)

            if isinstance(result, QueryParserSuccess):
                query = result.query
                inner_drc = query_inner_drc(query)
                # Validate: check for unquantified free variables on the
                # core DRC. The wrappers don't introduce new variables.
                validation_error = _validate_free_variables(inner_drc)
                if validation_error:
                    last_error = validation_error
                    print(f"⚠️ **Validation failed:** {validation_error}\n")
                    continue

                print(f"✅ **Parse succeeded**\n")
                pp = pretty_print_query(query)
                if isinstance(pp, PrintSuccess):
                    print(f"**Query (pretty):**\n```\n{pp.output}\n```\n")
                else:
                    pp_inner = pretty_print_indented(inner_drc)
                    if isinstance(pp_inner, PrintSuccess):
                        print(f"**DRC (pretty):**\n```\n{pp_inner.output}\n```\n")
                print(f"**Query (lisp):**\n```lisp\n{raw_lisp}\n```\n")

                # Decide whether the question implies SELECT DISTINCT.
                # Done after a successful parse so we don't waste an LLM
                # call on questions that won't produce SQL anyway.
                try:
                    distinct_decision = await decide_distinct(question, schema, config)
                    print(
                        f"**DISTINCT decision:** "
                        f"{'yes' if distinct_decision.use_distinct else 'no'}"
                        + (f" — {distinct_decision.reasoning}"
                           if distinct_decision.reasoning else "")
                        + "\n"
                    )
                except Exception as e:
                    print(f"⚠️ **DISTINCT decision failed:** {e} — defaulting to no DISTINCT\n")
                    from text_to_sql_planner.planner.llm_client import DistinctDecision
                    distinct_decision = DistinctDecision(
                        use_distinct=False,
                        reasoning=f"error: {e}",
                    )

                return ConversionSuccess(
                    expression=inner_drc,
                    query=query,
                    lisp_syntax=raw_lisp,
                    use_distinct=distinct_decision.use_distinct,
                    distinct_reasoning=distinct_decision.reasoning,
                )
            elif isinstance(result, QueryParserFailure):
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


def _validate_free_variables(expr: DRCExpression) -> str | None:
    """Check that no variables are free in the condition except result variables.

    Returns an error message if invalid, None if valid.
    """
    from text_to_sql_planner.types.drc import (
        ColumnVariable, AggregateVariable, QuantifierNode,
        LogicalConnectiveNode, NotNode, ComparisonNode,
        MembershipNode, ArithmeticNode, VariableRefNode,
        LiteralNode, FunctionCallNode,
    )

    # Collect result variable names (these are allowed to be free)
    result_var_names: set[str] = set()
    for rv in expr.result_variables:
        if isinstance(rv, ColumnVariable):
            result_var_names.add(rv.name)
        elif isinstance(rv, AggregateVariable):
            result_var_names.add(rv.column)

    # Collect all variables used in the condition
    all_vars: set[str] = set()
    _collect_vars(expr.condition, all_vars)

    # Collect all quantified (bound) variables
    bound_vars: set[str] = set()
    _collect_bound_vars(expr.condition, bound_vars)

    # Free variables = all_vars - bound_vars - result_var_names
    free_vars = all_vars - bound_vars - result_var_names

    if free_vars:
        return (
            f"Unquantified free variables: {sorted(free_vars)}. "
            f"These must be wrapped in (exists ...). "
            f"Result variables are: {sorted(result_var_names)}"
        )
    return None


def _collect_vars(node, vars_set: set[str]) -> None:
    """Collect all variable names used in a condition tree."""
    from text_to_sql_planner.types.drc import (
        QuantifierNode, LogicalConnectiveNode, NotNode, ComparisonNode,
        MembershipNode, ArithmeticNode, VariableRefNode, LiteralNode, FunctionCallNode,
    )

    if node is None:
        return
    if isinstance(node, VariableRefNode):
        vars_set.add(node.name)
    elif isinstance(node, MembershipNode):
        for v in node.variables:
            vars_set.add(v)
    elif isinstance(node, QuantifierNode):
        # Quantified vars are used but also bound
        for v in node.variables:
            vars_set.add(v)
        _collect_vars(node.body, vars_set)
    elif isinstance(node, LogicalConnectiveNode):
        _collect_vars(node.left, vars_set)
        _collect_vars(node.right, vars_set)
    elif isinstance(node, NotNode):
        _collect_vars(node.operand, vars_set)
    elif isinstance(node, ComparisonNode):
        _collect_vars(node.left, vars_set)
        _collect_vars(node.right, vars_set)
    elif isinstance(node, ArithmeticNode):
        _collect_vars(node.left, vars_set)
        _collect_vars(node.right, vars_set)
    elif isinstance(node, FunctionCallNode):
        for arg in node.arguments:
            _collect_vars(arg, vars_set)


def _collect_bound_vars(node, bound_set: set[str]) -> None:
    """Collect all variables that are bound by quantifiers."""
    from text_to_sql_planner.types.drc import (
        QuantifierNode, LogicalConnectiveNode, NotNode, ComparisonNode,
        MembershipNode, ArithmeticNode, FunctionCallNode,
    )

    if node is None:
        return
    if isinstance(node, QuantifierNode):
        for v in node.variables:
            bound_set.add(v)
        _collect_bound_vars(node.body, bound_set)
    elif isinstance(node, LogicalConnectiveNode):
        _collect_bound_vars(node.left, bound_set)
        _collect_bound_vars(node.right, bound_set)
    elif isinstance(node, NotNode):
        _collect_bound_vars(node.operand, bound_set)
    elif isinstance(node, ComparisonNode):
        _collect_bound_vars(node.left, bound_set)
        _collect_bound_vars(node.right, bound_set)
    elif isinstance(node, ArithmeticNode):
        _collect_bound_vars(node.left, bound_set)
        _collect_bound_vars(node.right, bound_set)
    elif isinstance(node, FunctionCallNode):
        for arg in node.arguments:
            _collect_bound_vars(arg, bound_set)
