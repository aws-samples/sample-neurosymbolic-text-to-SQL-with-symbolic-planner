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
                #
                # The hint is tailored to the apparent failure mode based
                # on the prior raw output: a generic hint about "exactly
                # N" nesting is actively misleading when the actual bug
                # is something else (e.g. a unary ``(and X)`` from a
                # join expression with one conjunct).
                hint = _retry_hint_for(last_raw_output, last_error)
                # The lead-in description should match the failure
                # mode: a parse error and a free-variable validation
                # error are different beasts and conflating them
                # actively misleads the next attempt.
                if "Unquantified free variables" in last_error:
                    lead = (
                        "[The previous attempt parsed but failed "
                        "validation. Validator: " + last_error + ". "
                    )
                else:
                    lead = (
                        "[The previous attempt produced syntactically "
                        "invalid DRC Lisp. Parser error: "
                        + last_error + ". "
                    )
                enhanced_question = (
                    f"{question}\n\n"
                    f"{lead}{hint} "
                    f"Produce a fresh, well-formed expression. Return "
                    f"ONLY the final S-expression with no surrounding "
                    f"prose, intermediate drafts, or 'let me redo' "
                    f"text.]"
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

            # Pull the LAST balanced top-level S-expression out of the
            # LLM output. The system prompt says "Return ONLY the DRC
            # expression in Lisp syntax, nothing else", but Claude
            # occasionally chains-of-thought in the response and emits
            # multiple candidates separated by prose ("Wait, let me
            # redo this more carefully."). Parsing the *first*
            # candidate then chokes on the trailing prose; parsing the
            # *last* matches the model's apparent final answer and
            # degrades to a no-op when the output is already a single
            # expression.
            parse_input = _extract_last_sexpr(raw_lisp)
            if parse_input != raw_lisp:
                print(
                    "ℹ️ **Multiple S-expressions detected in output;** "
                    "parsing the last one (the model's final answer).\n"
                )

            # Parse the Lisp syntax (extended DRC: optional limit / order-by
            # wrappers around a core ``(drc ...)`` form).
            result = parse_query(parse_input)

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
                    lisp_syntax=parse_input,
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


def _retry_hint_for(last_raw_output: str, last_error: str) -> str:
    """Tailor the retry hint to the apparent failure mode in
    ``last_raw_output``.

    The previous prompt always included a hint about "over-deep nesting
    in the negation of an 'exactly N' pattern", which is irrelevant
    (and actively misleading) for queries that don't involve
    negation-of-existential. This helper inspects the prior output and
    error to pick a more useful hint.

    Heuristics:

    * **Free-variable validator failure** — the parser succeeded but
      the framework's free-variable check rejected the result. Name
      the offending variables and tell the model to wrap them in an
      enclosing ``exists``. This is the most actionable hint we can
      produce; the variables are already extracted by
      :func:`_validate_free_variables` and live verbatim in the
      ``last_error`` message.
    * Unary ``(and X)`` / ``(or X)`` — the most common LLM mistake
      when joining tables with a single existential body. We accept
      this in the parser now, but for older error traces we still
      surface a hint so the next attempt avoids it.
    * Imbalanced parens — count opens vs closes; if they differ,
      surface that fact directly.
    * Negation of an existential — only when the output actually
      contains ``(not (exists`` do we mention exactly-N nesting.
    * Otherwise — generic balanced-parens hint.
    """
    err = last_error or ""
    text = last_raw_output or ""

    # The free-variable validator failure is the highest-signal
    # failure mode: we know exactly which variables the model forgot
    # to quantify. Pull them out of the error message and tell the
    # model to add them.
    free_vars = _free_vars_from_validator_error(err)
    if free_vars:
        names = ", ".join(free_vars)
        return (
            f"The previous expression parsed but had unquantified "
            f"variables: {names}. These appear inside a membership or "
            f"comparison but no enclosing (exists ...) binds them. Add "
            f"them to the variable list of the existing (exists ...) "
            f"that wraps their reference site (typically the inner "
            f"existential around the membership tuple), and double-"
            f"check that EVERY variable in every (in (...) Table) "
            f"tuple is either a result variable or bound by some "
            f"enclosing (exists ...)."
        )

    # Look for unary ``(and X)`` or ``(or X)``: a single sub-expression
    # between the operator and its matching close-paren. We use a
    # cheap depth-tracking textual scan rather than an AST scan
    # because the input failed to parse.
    if _has_unary_and_or(text):
        return (
            "The prior output contained a unary 'and' or 'or' "
            "expression like (and X) with a single operand. Use the "
            "operand directly instead of wrapping it; an existential "
            "with a single membership conjunct is just "
            "(in (...) Table) with no surrounding (and ...)."
        )

    # Balanced-paren check.
    opens = text.count("(")
    closes = text.count(")")
    if opens != closes:
        diff = closes - opens
        direction = (
            "more closes than opens" if diff > 0 else "more opens than closes"
        )
        return (
            f"The prior output had unbalanced parentheses "
            f"({direction}, off by {abs(diff)}). Re-count carefully."
        )

    # Negation-of-existential — surface the exactly-N hint only when
    # the output actually contains (not (exists ...).
    if "(not (exists" in text or "(not(exists" in text:
        return (
            "If the question involves an \"exactly N\" pattern, "
            "the negation should be a SINGLE (not (exists r_{N+1} ...)) "
            "with one extra witness pairwise-distinct from the prior N "
            "witnesses — not a chain of nested existentials inside the "
            "negation."
        )

    return "Re-balance the parentheses and re-emit the expression."


def _free_vars_from_validator_error(error_message: str) -> list[str]:
    """Extract the variable names from a free-variable validator error.

    The validator's error format is::

        Unquantified free variables: ['v1', 'v2', ...]. These must be ...

    We parse that bracketed Python-list literal directly with
    :func:`ast.literal_eval` so the extraction is unambiguous and
    requires no regex on lisp text. Returns ``[]`` if the message
    isn't from the free-variable validator or the variables list
    can't be safely parsed.
    """
    import ast

    marker = "Unquantified free variables: "
    idx = error_message.find(marker)
    if idx < 0:
        return []
    start = idx + len(marker)
    # The list literal extends from ``[`` to the matching ``]``;
    # walk the characters tracking bracket depth.
    if start >= len(error_message) or error_message[start] != "[":
        return []
    depth = 0
    end = start
    while end < len(error_message):
        ch = error_message[end]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    if depth != 0:
        return []
    snippet = error_message[start:end]
    try:
        parsed = ast.literal_eval(snippet)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str)]


def _extract_last_sexpr(text: str) -> str:
    """Return the last balanced top-level S-expression in ``text``.

    The LLM occasionally chains thought into its response, emitting
    multiple candidate expressions separated by prose:

        (drc (...) ...)

        Wait, let me redo this more carefully.

        (drc (...) ...)

    The system prompt says to return only the Lisp expression, but
    when Claude doesn't comply, parsing the first form succeeds and
    then chokes on the trailing prose. This helper walks the input
    once, depth-tracking parens with quote awareness, and returns
    the substring of the final balanced top-level form. When the
    output is already a single expression (or no balanced form is
    found at all), the original text is returned verbatim so this
    helper is a safe no-op on well-formed outputs.

    Quote awareness matters because DRC literals can contain
    unbalanced parens inside strings (``"foo)bar"``). We track
    double-quoted regions, with backslash-escape support, and ignore
    parens inside them.
    """
    if not text:
        return text

    # Walk the text once and record (start, end) of every balanced
    # top-level form (depth returns to 0 from 1).
    forms: list[tuple[int, int]] = []
    in_string = False
    escape = False
    depth = 0
    form_start = -1
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "(":
            if depth == 0:
                form_start = i
            depth += 1
        elif ch == ")":
            if depth > 0:
                depth -= 1
                if depth == 0 and form_start >= 0:
                    forms.append((form_start, i + 1))
                    form_start = -1
        i += 1

    if not forms:
        # No balanced form found — surface the input as-is so the
        # parser produces its usual error message.
        return text
    if len(forms) == 1:
        # Already a single top-level form; return verbatim to preserve
        # any leading whitespace / formatting the parser might use.
        return text

    last_start, last_end = forms[-1]
    return text[last_start:last_end]


def _has_unary_and_or(text: str) -> bool:
    """Detect ``(and X)`` or ``(or X)`` with exactly one immediate
    operand inside the matching parens.

    Walks the matching close-paren by tracking depth, counting top-level
    operands inside each ``(and …)`` / ``(or …)`` form. An "operand"
    is either a parenthesised sub-form or a contiguous run of non-paren
    non-space characters. Stops at the first match (we only need to
    know if any unary form exists).
    """
    for keyword in ("(and ", "(or "):
        i = 0
        while True:
            i = text.find(keyword, i)
            if i < 0:
                break
            j = i + len(keyword)
            depth = 1  # we're inside the outer (and / (or
            operand_count = 0
            in_word = False
            while j < len(text) and depth > 0:
                c = text[j]
                if c == "(":
                    if depth == 1:
                        operand_count += 1
                        in_word = False
                    depth += 1
                elif c == ")":
                    depth -= 1
                    in_word = False
                    if depth == 0:
                        break
                elif c.isspace():
                    in_word = False
                else:
                    if depth == 1 and not in_word:
                        operand_count += 1
                        in_word = True
                j += 1
            if depth == 0 and operand_count == 1:
                return True
            # Advance by one character so nested ``(and …)`` /
            # ``(or …)`` occurrences inside the just-scanned range
            # still get checked on subsequent iterations.
            i += 1
    return False


def _validate_free_variables(expr: DRCExpression) -> str | None:
    """Check that no variables are free in the condition except result variables.

    Performs a scope-aware free-variable analysis: a reference to ``v``
    is "free" at its position iff no enclosing quantifier binds ``v``.
    A variable that's bound by *some* quantifier in the tree but
    referenced *outside* that quantifier's scope is still free at the
    reference site — and that's a bug worth catching, because it means
    the LLM put ``(!= r3 r1)`` (or similar) outside the ``(exists r1
    ...)`` whose witness it intended to reference.

    Returns an error message if invalid, None if valid.
    """
    from text_to_sql_planner.types.drc import (
        ColumnVariable, AggregateVariable,
    )

    result_var_names: set[str] = set()
    for rv in expr.result_variables:
        if isinstance(rv, ColumnVariable):
            result_var_names.add(rv.name)
        elif isinstance(rv, AggregateVariable):
            result_var_names.add(rv.column)

    # Walk the condition with a running stack of currently-bound names.
    # Anything referenced while not in the stack and not a result
    # variable is genuinely free.
    free_unbound: set[str] = set()
    _collect_free_at_reference(expr.condition, frozenset(), result_var_names, free_unbound)

    if free_unbound:
        return (
            f"Unquantified free variables: {sorted(free_unbound)}. "
            f"These must be wrapped in (exists ...) whose scope spans "
            f"every reference site, or be result variables. "
            f"Result variables are: {sorted(result_var_names)}. "
            f"Common cause: a ``(not (exists ...))`` clause placed as a "
            f"sibling of the ``(exists r1 ...)`` whose witness it "
            f"references — the negation must be INSIDE the outer "
            f"existential, not next to it."
        )
    return None


def _collect_free_at_reference(
    node, bound_stack: frozenset, result_vars: set[str], free: set[str],
) -> None:
    """Walk the AST tracking the current bound-name set.

    A reference is added to ``free`` iff its name is not in
    ``bound_stack`` and not in ``result_vars``.
    """
    from text_to_sql_planner.types.drc import (
        QuantifierNode, LogicalConnectiveNode, NotNode, ComparisonNode,
        MembershipNode, ArithmeticNode, VariableRefNode, FunctionCallNode,
    )

    if node is None:
        return
    if isinstance(node, VariableRefNode):
        if node.name not in bound_stack and node.name not in result_vars:
            free.add(node.name)
        return
    if isinstance(node, MembershipNode):
        for v in node.variables:
            if v not in bound_stack and v not in result_vars:
                free.add(v)
        return
    if isinstance(node, QuantifierNode):
        new_bound = bound_stack | set(node.variables)
        _collect_free_at_reference(node.body, new_bound, result_vars, free)
        return
    if isinstance(node, LogicalConnectiveNode):
        _collect_free_at_reference(node.left, bound_stack, result_vars, free)
        _collect_free_at_reference(node.right, bound_stack, result_vars, free)
        return
    if isinstance(node, NotNode):
        _collect_free_at_reference(node.operand, bound_stack, result_vars, free)
        return
    if isinstance(node, ComparisonNode):
        _collect_free_at_reference(node.left, bound_stack, result_vars, free)
        _collect_free_at_reference(node.right, bound_stack, result_vars, free)
        return
    if isinstance(node, ArithmeticNode):
        _collect_free_at_reference(node.left, bound_stack, result_vars, free)
        _collect_free_at_reference(node.right, bound_stack, result_vars, free)
        return
    if isinstance(node, FunctionCallNode):
        for a in node.arguments:
            _collect_free_at_reference(a, bound_stack, result_vars, free)
        return


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
