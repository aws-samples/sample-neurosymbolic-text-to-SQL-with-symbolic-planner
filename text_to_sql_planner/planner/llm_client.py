"""LLM client: async wrapper around AWS Bedrock (boto3) for Claude."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from text_to_sql_planner.types.drc import DRCExpression

try:
    import boto3
except ImportError:
    boto3 = None  # type: ignore


@dataclass
class LLMClientConfig:
    """Configuration for the LLM client."""

    region: str = "us-east-1"
    model_id: str = "global.anthropic.claude-opus-4-6-v1"
    max_tokens: int = 4096


@dataclass
class OperatorSelection:
    """Result of asking the LLM to select an operator."""

    operator: str  # RAOperatorType
    input_indices: list[int]  # indices into available relations
    params: dict  # raw params to be converted to OperatorParams
    reasoning: str


@dataclass
class IntermediateRelation:
    """An intermediate relation produced during planning."""

    index: int
    expression: DRCExpression
    columns: list[str]
    summary: str = ""  # one-sentence natural language description


@dataclass
class RejectedProposal:
    """A previously-attempted operator selection that the planner rejected.

    Surfaced back to the LLM on retry so it doesn't loop on the same
    proposal. Carries enough information to make a counter-proposal:
    which operator on which inputs with which params, and *why* it was
    rejected.
    """

    operator: str  # RAOperatorType
    input_indices: list[int]
    params: dict
    reason: str  # short human-readable rejection reason


def _get_bedrock_client(config: LLMClientConfig):
    """Create a Bedrock Runtime client."""
    if boto3 is None:
        raise ImportError(
            "The 'boto3' package is required but not installed. "
            "Install it with: pip install boto3"
        )
    return boto3.client(
        "bedrock-runtime",
        region_name=config.region,
    )


_SYSTEM_PROMPT_OPERATOR = """You are a relational algebra planner. You select operators to transform base table relations into a target DRC expression.

Available operators:
- selection: Filter rows using a condition. Requires 1 input.
- projection: Select specific columns. Requires 1 input.
- rename: Rename one or more columns of a relation. Requires 1 input. Use this BEFORE a self-join to disambiguate columns that would otherwise collide.
- aggregate: Promote the input's single result variable from a column to an aggregate (COUNT/SUM/AVG/MIN/MAX). Requires 1 input. The input MUST already have exactly one column-typed result variable, and that column's name MUST match the ``column`` parameter. Use this when the target DRC's result variable is an aggregate like ``(SUM x)`` and you've already built the underlying ``{x | …}`` relation — ``rename`` cannot perform this transition because it can't change a result variable's *kind*.
- join: Natural join on shared columns. Requires 2 inputs.
- cartesian_product: Cross product of two relations. Requires 2 inputs.
- union: Set union of two compatible relations. Requires 2 inputs.
- difference: Set difference R minus S; tuples in R that are not in S. Requires 2 inputs of the same arity.
- division: Relational division. Requires 2 inputs.

DRC Notation Guide:
- Expressions use Lisp S-expression syntax
- (drc (result-vars) condition) - top-level expression
- (in (vars...) RelationName) - membership
- (and cond1 cond2), (or cond1 cond2), (not cond) - logical
- (= x y), (!= x y), (< x y), (> x y), (<= x y), (>= x y) - comparison
- (exists (vars...) body) - existential quantifier (use (in ...) in body to bind to a relation)
- (forall (vars...) body) - universal quantifier (use (in ...) in body to bind to a relation)

CRITICAL RULES — pick the right binary operator:

1. **Prefer ``join`` over ``cartesian_product`` whenever the operands share a column whose values you want to be equal.**
   ``join`` on ``[col]`` is equivalent to ``cartesian_product`` followed by ``selection WHERE T1.col = T2.col``, but it does the equality
   in one step and keeps the tuple-binding tight. ``cartesian_product`` on operands that share a column name produces *renamed*
   ``col_1`` / ``col_2`` outputs that are NOT bound together. If you then drop one of them via ``projection``, the binding is GONE
   and the resulting relation no longer correctly represents the intended condition.

2. **For self-joins (joining a relation with itself), use ``rename`` first to disambiguate.**
   Standard pattern for "≥N reviews per employee":
       a. Start with Performance_Reviews (or a projection thereof to ``review_id, emp_id``).
       b. ``rename`` it to give one copy distinct review-id columns: ``rename {review_id: review_id_2}``.
       c. ``join`` the original with the renamed copy on ``[emp_id]``. This produces pairs of reviews for the same employee
          with two distinct review-id columns side by side, ready for a ``selection`` that requires ``review_id != review_id_2``.
       d. Repeat for the third witness if needed (rename the next copy to ``review_id_3``, join, select distinctness).
   Avoid ``cartesian_product`` of Performance_Reviews with itself — the shared ``emp_id`` column gets renamed to ``emp_id_1`` /
   ``emp_id_2`` which BREAKS the binding, and recovering it via projection is unsound.

3. **For "exactly N" patterns, use ``difference``: (≥N reviews) − (≥(N+1) reviews).**
   Build the ≥N and ≥(N+1) relations by following rule (2). Once both have the same column shape (typically just ``emp_id``),
   ``difference`` of them gives "exactly N". Then ``join`` with Employees on ``emp_id`` to attach names.

4. **NEVER drop a binding column via ``projection`` when its values must remain tied to another column's values for correctness.**
   Specifically: after ``cartesian_product`` of two relations that share a column, the projection MUST either keep both copies
   (with distinct names) or be preceded by a ``selection`` that asserts the equality. If you skip the selection and just project,
   the resulting relation is logically a "there exists *some* tuple" check, not "*this* tuple" — and equivalence with the target
   will fail for non-trivial reasons.

Select the next operator to apply to move closer to the target expression.
Provide your reasoning, the operator type, which input relations to use (by index), and the operator parameters."""


_OPERATOR_TOOL = {
    "name": "select_operator",
    "description": "Select the next relational algebra operator to apply.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Step-by-step reasoning for why this operator is chosen.",
            },
            "operator": {
                "type": "string",
                "enum": [
                    "selection",
                    "join",
                    "projection",
                    "rename",
                    "aggregate",
                    "ratio",
                    "cartesian_product",
                    "union",
                    "difference",
                    "division",
                ],
                "description": "The operator to apply.",
            },
            "input_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Indices of the relations to use as inputs (0-based).",
            },
            "params": {
                "type": "object",
                "description": "Operator-specific parameters. For selection: {condition: <lisp-string>}. For projection: {columns: [col1, col2]}. For join: {join_columns: [col]}. For rename: {mapping: {old_name: new_name, ...}}. For aggregate: {function: 'COUNT'|'SUM'|'AVG'|'MIN'|'MAX', column: <name-of-input's-single-column>}. For ratio: {operator: '/'|'-'|'+'|'*', numerator_function: 'COUNT', numerator_column: 'col1', denominator_function: 'COUNT', denominator_column: 'col2', numerator_condition: '<optional lisp condition for COUNT_IF on numerator>', denominator_condition: '<optional lisp condition for COUNT_IF on denominator>', scalar_multiplier: <optional number like 100 for percentage>}. For cartesian_product/union/difference/division: {}.",
            },
        },
        "required": ["reasoning", "operator", "input_indices", "params"],
    },
}


async def select_operator(
    table_relations: list[dict],
    intermediate_relations: list[IntermediateRelation],
    target_relation: str,
    temperature: float,
    config: LLMClientConfig | None = None,
    rejected_proposals: list[RejectedProposal] | None = None,
) -> OperatorSelection:
    """Ask the LLM to select the next operator to apply.

    Uses AWS Bedrock with boto3 to invoke Claude. Authenticates via
    standard AWS credentials (environment variables, IAM role, etc.).

    Args:
        table_relations: List of dicts with table_name, columns, lisp_syntax.
        intermediate_relations: Previously computed intermediate relations.
        target_relation: The target DRC expression in Lisp syntax.
        temperature: Sampling temperature.
        config: LLM client configuration.
        rejected_proposals: Operator picks already attempted within the
            current iteration. The model is shown these and instructed
            not to repeat them. Defaults to ``None`` (empty).

    Returns:
        OperatorSelection with the chosen operator and parameters.

    Raises:
        ImportError: If boto3 is not installed.
        RuntimeError: If the LLM fails to return a valid tool use.
    """
    if config is None:
        config = LLMClientConfig()

    client = _get_bedrock_client(config)

    # Build the user message describing available relations
    relations_desc = "Available relations:\n"
    for i, tr in enumerate(table_relations):
        summary = tr.get('summary', f"All rows from table {tr['table_name']}")
        relations_desc += f"  [{i}] Table '{tr['table_name']}' columns={tr['columns']}\n"
        relations_desc += f"      Summary: {summary}\n"
        relations_desc += f"      DRC: {tr['lisp_syntax']}\n"

    for ir in intermediate_relations:
        idx = ir.index
        ir_lisp = ""
        from text_to_sql_planner.printer import print_lisp as _print_lisp, PrintSuccess as _PS
        lisp_result = _print_lisp(ir.expression)
        if isinstance(lisp_result, _PS):
            ir_lisp = lisp_result.output
        relations_desc += f"  [{idx}] Intermediate columns={ir.columns}\n"
        relations_desc += f"      Summary: {ir.summary}\n"
        if ir_lisp:
            relations_desc += f"      DRC: {ir_lisp}\n"

    relations_desc += f"\nTarget expression:\n  {target_relation}\n"
    relations_desc += (
        "\nIMPORTANT: Do NOT produce a relation that is identical to any of the "
        "existing relations listed above. Each new operator application must produce "
        "a genuinely new result that moves closer to the target.\n"
    )

    if rejected_proposals:
        relations_desc += (
            "\nREJECTED PROPOSALS — these operator picks were already attempted "
            "in this iteration and rejected. DO NOT repeat any of them; pick a "
            "different operator, different input indices, or different params.\n"
        )
        for i, rp in enumerate(rejected_proposals, start=1):
            relations_desc += (
                f"  [{i}] {rp.operator} inputs={rp.input_indices} "
                f"params={rp.params} — rejected: {rp.reason}\n"
            )

    relations_desc += "\nSelect the next operator to apply."

    # Build the Bedrock converse request
    request_body = {
        "modelId": config.model_id,
        "system": [{"text": _SYSTEM_PROMPT_OPERATOR}],
        "messages": [
            {"role": "user", "content": [{"text": relations_desc}]}
        ],
        "inferenceConfig": {
            "maxTokens": config.max_tokens,
            "temperature": temperature,
        },
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": _OPERATOR_TOOL["name"],
                        "description": _OPERATOR_TOOL["description"],
                        "inputSchema": {"json": _OPERATOR_TOOL["input_schema"]},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "select_operator"}},
        },
    }

    response = client.converse(**request_body)

    # Extract tool use from response
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    for block in content_blocks:
        if block.get("toolUse") and block["toolUse"].get("name") == "select_operator":
            args = block["toolUse"]["input"]
            return OperatorSelection(
                operator=args["operator"],
                input_indices=args["input_indices"],
                params=args.get("params", {}),
                reasoning=args.get("reasoning", ""),
            )

    raise RuntimeError("LLM did not return a valid tool use response for operator selection.")


_SYSTEM_PROMPT_DRC = """You are a formal logic expert. Convert natural language questions about a database into Domain Relational Calculus (DRC) expressions using Lisp S-expression syntax.

Syntax rules:
- Top-level: (drc (result-var1 result-var2 ...) condition)
- Result variables can be plain column names OR aggregate functions:
  - Plain column: just the name, e.g. name, id, age
  - Aggregate: (COUNT col), (COUNT_DISTINCT col), (SUM col), (AVG col), (MIN col), (MAX col)
    Use (COUNT_DISTINCT col) when the question or evidence says "distinct count" or "number of distinct".
  - Conditional aggregate: (COUNT_IF condition col) — counts col only where condition is true. Use for "what percentage of X satisfy Y" patterns: (/ (COUNT_IF (= status "A") id) (COUNT id)).
  - Conditional output: (IF condition "then_value" "else_value") — produces IIF(condition, 'then', 'else') in SQL. Use for yes/no questions: "Is X true?" → (IF (condition) "Yes" "No"). Also for categorization: (IF (is-not-null col) "has value" "no value").
  - Arithmetic of aggregates: (/ (COUNT col1) (COUNT col2)), (* (SUM col1) (AVG col2)), etc.
    Use this for ratio/percentage questions like "average number of X per Y" = (/ (COUNT x) (COUNT y)).
- Membership: (in (var1 var2 ...) TableName) — asserts that the tuple (var1, var2, ...) is a row in TableName
- Logical: (and cond1 cond2), (or cond1 cond2), (not cond), (implies cond1 cond2)
  - ``and`` and ``or`` accept TWO OR MORE operands. NEVER write a unary ``(and X)`` or ``(or X)`` — if there is only one conjunct/disjunct, write the operand directly with no surrounding ``(and ...)`` or ``(or ...)``.
- Comparison: (= x y), (!= x y), (< x y), (> x y), (<= x y), (>= x y)
- Quantifiers: (exists (var1 var2 ...) body), (forall (var1 var2 ...) body)
  - Quantifiers just bind variables. Use (in ...) inside the body to constrain them to a relation.
- Date functions: CURRENT_DATE (today's date), (DATE_SUB expr days) (subtract days from a date), (DATEDIFF expr1 expr2) (age in years between two dates — e.g., ``(DATEDIFF CURRENT_DATE Birthday)`` for current age). Use DATEDIFF for "older than N years" patterns: ``(> (DATEDIFF CURRENT_DATE Birthday) N)``.
- String pattern matching: (LIKE column "pattern") — SQL LIKE semantics. Use ``%`` as a wildcard in the pattern. ``"%data%"`` matches any string containing ``data``; ``"data%"`` matches a prefix; ``"%data"`` matches a suffix.
- NULL check: (is-not-null column) — SQL ``column IS NOT NULL``. Use this whenever the evidence or question says "is not null" or "is not empty" or "has a value". Do NOT use ``(!= column "")`` or ``(not (= column NULL))`` — those have wrong SQL NULL semantics.
- Literals: strings in double quotes "hello", numbers as-is 42
- Variables: plain identifiers like name, age, id

IMPORTANT RULES:
- Result variables are FREE variables — they must NOT appear as quantified variables.
- CRITICAL: Every variable in a membership (in ...) that is NOT a result variable and NOT the column being aggregated MUST be wrapped in an existential quantifier (exists ...). There must be NO free variables in the condition other than the result variables (and aggregate columns). If a table has 10 columns but you only need 2 as result variables, the other 8 MUST be existentially quantified.
- When the question asks "how many", "count", "total number of", etc., use (COUNT col) in the result variables.
- For age/date calculations, use CURRENT_DATE and (DATE_SUB CURRENT_DATE days). For example, "at least 30 years old" means (>= (DATE_SUB CURRENT_DATE 10950) date_of_birth) where 10950 = 30*365.
- Quantifiers bind variables; membership (in) constrains them to a table. Always pair them.
- When "Evidence" is provided after the question, it contains AUTHORITATIVE column-to-concept mappings. ALWAYS follow these mappings exactly — they override any other interpretation of the question. For example, if evidence says "X refers to column = 'value'", use ONLY that column/value in your DRC, even if the English phrasing suggests something else.
- When evidence provides an inequality operator like ``column != 'value'`` or ``column <> 'value'``, use it as a direct comparison ``(!= column "value")`` in the condition — do NOT convert it into a ``(not (exists ...))`` pattern. The evidence's operator choice is intentional.
- String comparisons in SQLite are CASE-SENSITIVE for ``=``. If the evidence gives a value like ``'cryokinesis'`` but the schema likely stores it with different casing (e.g., ``'Cryokinesis'``), use ``LIKE`` with the pattern instead of ``=`` for robustness: ``(LIKE column "cryokinesis")``. This is especially important for proper nouns that might be capitalized in the database.
- When joining tables, use ONLY the foreign-key column specified in the schema or evidence. Do NOT join on multiple alternative columns with OR. If the evidence says "player refers to player_name = 'X'" without specifying a join column, look at the schema's column names to identify the single correct FK relationship (e.g., ``player_api_id``). A join should be a single equality ``(= t1.fk_col t2.pk_col)``, never a disjunction of columns.

Examples:

Question: "Find all employees in department 5"
Schema: CREATE TABLE employees (id INT, name VARCHAR, dept_id INT)
Answer: (drc (id name dept_id) (and (in (id name dept_id) employees) (= dept_id 5)))
Note: To filter a column on a literal value, ALWAYS bind the column to a variable inside the (in (...) Table) tuple, then constrain it with (= variable literal) as a separate conjunct. Literals (numbers, strings) MUST NEVER appear directly in the (in (...) Table) tuple itself — that position takes only variable names. The same rule applies whether the literal is a number (= weight_kg 169) or a string (= name "Alice").

Question: "Find the race of the superhero who weighed 169 kg"
Schema: CREATE TABLE superhero (id INT, name VARCHAR, race_id INT, weight_kg INT); CREATE TABLE race (id INT, race VARCHAR)
Answer: (drc (race) (exists (rid) (and (in (rid race) race) (exists (hid sname weight_kg) (and (in (hid sname rid weight_kg) superhero) (= weight_kg 169))))))
Note: ``weight_kg`` is bound to a variable in the membership, not written as ``169`` directly. The constant goes in a separate ``(= weight_kg 169)`` conjunct.

Question: "Find employee names in department 5"
Schema: CREATE TABLE employees (id INT, name VARCHAR, dept_id INT)
Answer: (drc (name) (exists (id dept_id) (and (in (id name dept_id) employees) (= dept_id 5))))
Note: id and dept_id are NOT result variables, so they MUST be existentially quantified.

Question: "How many students are enrolled in CS courses?"
Schema: CREATE TABLE Students (s_id INT, name VARCHAR); CREATE TABLE Enrolled (s_id INT, c_id INT); CREATE TABLE Courses (c_id INT, c_type VARCHAR)
Answer: (drc ((COUNT s_id)) (exists (name) (and (in (s_id name) Students) (exists (es ec) (and (in (es ec) Enrolled) (= es s_id) (exists (cc ctype) (and (in (cc ctype) Courses) (= cc ec) (= ctype "Computer Science"))))))))

Question: "What is the average grade of students in course 101?"
Schema: CREATE TABLE Enrolled (s_id INT, c_id INT, grade INT)
Answer: (drc ((AVG grade)) (exists (s_id c_id) (and (in (s_id c_id grade) Enrolled) (= c_id 101))))

Question: "Find employees who have no performance reviews"
Schema: CREATE TABLE Employees (emp_id INT, first_name VARCHAR, last_name VARCHAR, dept_id INT); CREATE TABLE Reviews (review_id INT, emp_id INT, rating INT)
Answer: (drc (emp_id first_name last_name) (exists (dept_id) (and (in (emp_id first_name last_name dept_id) Employees) (not (exists (review_id rating) (in (review_id emp_id rating) Reviews))))))
Note: dept_id is NOT a result variable so it's quantified. emp_id IS a result variable so it stays free.

Question: "Show employee names and salaries"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR); CREATE TABLE Compensation (comp_id INT, emp_id INT, salary DECIMAL)
Answer: (drc (name salary) (exists (emp_id) (and (in (emp_id name) Employees) (exists (comp_id) (in (comp_id emp_id salary) Compensation)))))
Note: The inner existential body has only ONE conjunct — the (in ...) membership — so it appears bare without a surrounding (and ...). Writing (exists (comp_id) (and (in (...) Compensation))) with a unary 'and' is INVALID — use the membership directly.

Cardinality patterns ("at least N", "two or more", "more than N", "exactly N"):
- "at least N" / "N or more" / "two or more" → assert the existence of N distinct witnesses, joined by AND, all pairwise distinct. DO NOT add ``(not (exists ...))`` for an extra witness — that turns "≥N" into "exactly N" or "≥N+1".
- "more than N" → same as "at least N+1".
- "exactly N" → assert N distinct witnesses AND ``(not (exists ...))`` for a SINGLE (N+1)-th witness distinct from all the prior ones.
- Each witness must be wrapped in its own existential quantifier and constrained by an (in ...) membership.
- Witnesses must be pairwise distinct via (!= ...) on the identifying columns.
- CRITICAL: The witness pattern is ONLY practical for N ≤ 3. For N > 3 (e.g., "more than 5", "at least 10", "over 15"), DO NOT use the witness pattern — it creates N self-joins with O(N²) inequality checks that produce unusable SQL. Instead, express the cardinality constraint as a GROUP BY + COUNT aggregate: project the grouping key and the counted column, then use the ratio operator or aggregate to compute COUNT, and filter via the condition. For example, "employees with more than 5 reviews" → project (emp_id, review_id) from the join, then use aggregate COUNT with a selection condition.

CRITICAL — the negation in "exactly N" is ONE existential, NOT a chain.
The shape is always:
    (not (exists (r_{N+1} ...) (and (in (...) Reviews) (and (!= r_{N+1} r_1) ... (!= r_{N+1} r_N)))))
with exactly one (exists ...) inside (not ...) and a flat AND-chain of (!= r_{N+1} r_k) inequalities — one per prior witness. Do NOT introduce r_{N+2}, r_{N+3}, etc. inside the negation. Doing so changes the meaning ("not ≥ N+M" instead of "not ≥ N+1") and produces deeply nested parentheses that are easy to miscount.

Worked example for N=2 — the right shape:
    (exists (r1 ...) (and PR(r1, ...)
      (exists (r2 ...) (and PR(r2, ...)
        (and (!= r1 r2)
             (not (exists (r3 ...) (and PR(r3, ...) (and (!= r3 r1) (!= r3 r2))))))))))

Wrong shape (do NOT do this):
    (not (exists (r3 ...) (and PR(r3, ...) (exists (r4 ...) ... (exists (r5 ...) ...)))))
This says "not ≥ 5", not "not ≥ 3", and is the most common failure mode.

Question: "List all employees that have two or more performance reviews"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR); CREATE TABLE Reviews (review_id INT, emp_id INT, rating INT)
Answer: (drc (emp_id name) (and (in (emp_id name) Employees) (exists (r1 rating1) (and (in (r1 emp_id rating1) Reviews) (exists (r2 rating2) (and (in (r2 emp_id rating2) Reviews) (!= r1 r2)))))))
Note: TWO existential witnesses (r1 and r2), both bound to ``emp_id``, with ``(!= r1 r2)``. NO third witness, NO ``(not (exists ...))`` clause.

Question: "List employees with exactly two performance reviews"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR); CREATE TABLE Reviews (review_id INT, emp_id INT, rating INT)
Answer: (drc (emp_id name) (and (in (emp_id name) Employees) (exists (r1 rating1) (and (in (r1 emp_id rating1) Reviews) (exists (r2 rating2) (and (in (r2 emp_id rating2) Reviews) (and (!= r1 r2) (not (exists (r3 rating3) (and (in (r3 emp_id rating3) Reviews) (and (!= r3 r1) (!= r3 r2))))))))))))
Note: TWO positive witnesses AND a (not (exists ...)) for a third one. CRITICAL: ``r1`` and ``r2`` MUST be bound by ``(exists ...)`` whose scope spans BOTH the positive ``(in ...)`` memberships AND the negated ``(not (exists ...))`` clause. Putting the negation as a SIBLING of the positive witnesses (i.e. outside the ``(exists r1 ...)`` and ``(exists r2 ...)`` scopes) makes the ``r1``/``r2`` references inside the negation refer to free constants instead of the chosen witnesses, which silently breaks the formula.

Question: "List all employees that have three or more performance reviews"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR); CREATE TABLE Reviews (review_id INT, emp_id INT, rating INT)
Answer: (drc (emp_id name) (and (in (emp_id name) Employees) (exists (r1 rating1) (and (in (r1 emp_id rating1) Reviews) (exists (r2 rating2) (and (in (r2 emp_id rating2) Reviews) (and (!= r1 r2) (exists (r3 rating3) (and (in (r3 emp_id rating3) Reviews) (and (!= r3 r1) (!= r3 r2))))))))))
Note: THREE positive witnesses, all pairwise distinct (r1≠r2, r3≠r1, r3≠r2). NO ``(not (exists ...))``.

Ordering and limiting (extended DRC):
- DRC itself is set-based — it has no row order or row count. To express "top N", "first N by ...", "the highest/lowest", "ranked by ...", etc., wrap the core DRC in non-relational operators:
    (limit N (order-by ((key dir) (key2 dir2) ...) (drc (...) ...)))
- ``key`` is either a bare column name from the result variables, or an aggregate sub-form ``(AGG col)`` mirroring an aggregate result variable.
- ``dir`` is ``asc`` or ``desc``.
- ``order-by`` is the inner wrapper, ``limit`` is outermost. Either may be omitted; LIMIT without ORDER BY is non-deterministic and should usually be avoided.
- The wrappers DO NOT change the inner ``(drc ...)`` — keep result variables and condition exactly as they would be without the wrappers.

Question: "Show me the top 5 most-compensated employees"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR, salary DECIMAL)
Answer: (limit 5 (order-by ((salary desc)) (drc (emp_id name salary) (in (emp_id name salary) Employees))))

Question: "List the 10 oldest employees by hire date"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR, hire_date DATE)
Answer: (limit 10 (order-by ((hire_date asc)) (drc (emp_id name hire_date) (in (emp_id name hire_date) Employees))))

Question: "Which 3 departments have the most employees?"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR, dept_id INT); CREATE TABLE Departments (dept_id INT, dept_name VARCHAR)
Answer: (limit 3 (order-by (((COUNT emp_id) desc)) (drc (dept_name) (exists (dept_id) (and (in (dept_id dept_name) Departments) (exists (emp_id name) (and (in (emp_id name dept_id) Employees))))))))
Note: This is a ranking question ("which … most …"), so the result variables contain ONLY ``dept_name`` — the entity being ranked. ``(COUNT emp_id)`` appears in the order-by key but NOT in the result variables. See the "Ranking questions" section below.

Ranking questions ("which X has the highest/lowest Y", "what is the X with the most Z", "find the top X by Y"):
- These questions name TWO things: the entity to identify (X) and the criterion to rank by (Y / Z).
- The result variables MUST contain ONLY X — the identifying columns. The ranking criterion goes in the ``order-by`` key, NOT in the result variables.
- This convention matches how the question is phrased in English: "which gas station has the highest revenue" asks for *the gas station*, not *(gas station, revenue)*. Adding the ranking column to the SELECT list answers a different question ("show me each gas station with its revenue, then pick the top one").
- The ranking criterion can still be an aggregate in the ``order-by`` key — that's fine, aggregate keys don't have to mirror result variables.
- This rule applies whenever the question uses superlative or ranking phrasing: "highest", "lowest", "most", "least", "top", "bottom", "biggest", "smallest", "first", "last by …".
- CRITICAL: NEVER use ``forall`` to express "the row with the maximum/minimum value". Instead, ALWAYS use ``(limit 1 (order-by ((col desc)) ...))``. The ``forall`` approach (asserting "for all other rows, this row's value >= theirs") is logically equivalent but MUCH harder for the planner to build. The ``order-by + limit`` approach maps directly to SQL ``ORDER BY col DESC LIMIT 1`` and is trivial to construct.

Question: "Which gas station has the highest amount of revenue?"
Schema: CREATE TABLE Transactions (TransactionID INT, GasStationID INT, Price REAL)
Answer: (limit 1 (order-by (((SUM Price) desc)) (drc (GasStationID) (exists (TransactionID Price) (in (TransactionID GasStationID Price) Transactions)))))
Note: Result variables are JUST ``GasStationID`` — the ranking column ``(SUM Price)`` appears only in the order-by key. Putting ``(SUM Price)`` in the result variables would project two columns where the question asks for one.

Question: "What is the name of the employee with the highest salary?"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR, salary DECIMAL)
Answer: (limit 1 (order-by ((salary desc)) (drc (name) (exists (emp_id salary) (in (emp_id name salary) Employees)))))
Note: Only ``name`` is requested. ``salary`` is the ranking criterion and stays in order-by; ``emp_id`` and ``salary`` are existentially quantified.

Question: "Show the top 5 employees and their salaries"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR, salary DECIMAL)
Answer: (limit 5 (order-by ((salary desc)) (drc (name salary) (exists (emp_id) (in (emp_id name salary) Employees)))))
Note: Contrast with the previous example. Here the question explicitly asks for BOTH name AND salary ("employees AND their salaries"), so both appear in the result variables. The trigger words for the multi-column projection are conjunctions ("and", "with their", "along with") between the entity and the metric.

Ratio / percentage / average-per-entity patterns:
- "What is the average number of X per Y" → divide COUNT(X) by COUNT(DISTINCT Y): ``(/ (COUNT x) (COUNT y))``
- "What percentage of X satisfy condition C" → divide conditional count by total count: ``(/ (COUNT_IF condition col) (COUNT col))``
  IMPORTANT: The base population filter (X) goes in the membership/condition body (it restricts which rows are counted in BOTH numerator and denominator). Only the NUMERATOR condition (C) goes in the COUNT_IF. Example: "What percentage of players under 180cm have rating > 70?" → filter to height < 180 in the condition body, then ``(/ (COUNT_IF (> rating 70) id) (COUNT id))``. Do NOT put ALL conditions inside COUNT_IF — that changes the denominator.
- ``(COUNT_IF condition col)`` counts ``col`` only for rows where ``condition`` is true. It maps to SQL ``COUNT(CASE WHEN condition THEN col END)``.
- The ``/`` operator in result-variable position produces SQL ``AGG1(...) / AGG2(...)`` in a single SELECT.

Question: "What is the average number of badges per user with over 200 views?"
Schema: CREATE TABLE badges (Id INT, UserId INT, Name TEXT); CREATE TABLE users (Id INT, Views INT, DisplayName TEXT)
Answer: (drc ((/ (COUNT Id) (COUNT DisplayName))) (exists (UserId Name) (and (in (Id UserId Name) badges) (exists (uid Views) (and (in (uid Views DisplayName) users) (= uid UserId) (> Views 200))))))
Note: The result is a ratio of two aggregates: total badge count divided by distinct user count. Both aggregates share the same underlying condition (users with Views > 200 who have badges).

Question: "What percentage of patients with high GOT levels are diagnosed with SLE?"
Schema: CREATE TABLE Patient (ID INT, Diagnosis TEXT); CREATE TABLE Laboratory (ID INT, GOT REAL)
Answer: (drc ((/ (COUNT_IF (LIKE Diagnosis "%SLE%") ID) (COUNT ID))) (exists (GOT) (and (in (ID GOT) Laboratory) (>= GOT 60) (exists (Diagnosis) (in (ID Diagnosis) Patient)))))
Note: ``COUNT_IF`` counts only rows where the condition (Diagnosis contains "SLE") is true. The denominator ``(COUNT ID)`` counts all rows matching the base filter (GOT >= 60). The ratio gives the percentage.

IMPORTANT — percentage multiplication:
- When the evidence or question says "percentage" and the evidence formula explicitly includes ``* 100``, you MUST wrap the ratio in ``(* ... 100)`` to produce a 0–100 value. Example: ``(* (/ (COUNT_IF condition col) (COUNT col)) 100)``
- When the evidence gives a formula like ``DIVIDE(X, Y) * 100``, replicate that EXACTLY as ``(* (/ X Y) 100)`` in the arithmetic aggregate result variable.
- A bare ``(/ (COUNT_IF ...) (COUNT ...))`` produces a 0-to-1 decimal, NOT a percentage. Only omit ``* 100`` if the evidence omits it.

IMPORTANT — computed result variables:
- When evidence defines a value as a formula (e.g., "eligible rate = A / B"), the RESULT VARIABLE must be that formula ``(/ A B)``, NOT the raw components ``A B`` as separate columns.
- WRONG: ``(drc (A B) ...)`` then ordering by ``(/ A B)`` — this returns raw columns, not the computed value.
- RIGHT: ``(drc ((/ A B)) ...)`` — the arithmetic IS the result variable.
- If the question asks "list the rates" and evidence says "rate = X / Y", the DRC result variable is ``(/ X Y)`` as a single arithmetic expression.

Return ONLY the DRC expression in Lisp syntax, nothing else."""


async def convert_question_to_drc(
    question: str,
    schema: str,
    config: LLMClientConfig | None = None,
) -> str:
    """Ask the LLM to convert a natural language question into DRC Lisp syntax.

    Uses AWS Bedrock with boto3 to invoke Claude. Authenticates via
    standard AWS credentials (environment variables, IAM role, etc.).

    Args:
        question: The natural language question.
        schema: The database schema (CREATE TABLE statements).
        config: LLM client configuration.

    Returns:
        The raw Lisp syntax string from the LLM.

    Raises:
        ImportError: If boto3 is not installed.
        RuntimeError: If the LLM fails to return a response.
    """
    if config is None:
        config = LLMClientConfig()

    client = _get_bedrock_client(config)

    user_message = f"Schema:\n{schema}\n\nQuestion: {question}"

    request_body = {
        "modelId": config.model_id,
        "system": [{"text": _SYSTEM_PROMPT_DRC}],
        "messages": [
            {"role": "user", "content": [{"text": user_message}]}
        ],
        "inferenceConfig": {
            "maxTokens": config.max_tokens,
            "temperature": 0.0,
        },
    }

    response = client.converse(**request_body)

    # Extract text from response
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    for block in content_blocks:
        if "text" in block:
            return block["text"].strip()

    raise RuntimeError("LLM did not return a text response for DRC conversion.")


_SYSTEM_PROMPT_SUMMARIZE = """Summarize the given DRC (Domain Relational Calculus) expression in one short sentence describing what data it represents in plain English. Be concise — one sentence only, no more than 15 words. Do not include technical notation.

Examples:
- "All students" → "All rows from the Students table"
- "{name | name,age ∈ Students ∧ age > 21}" → "Names of students older than 21"
- "{s_id | s_id,c_id ∈ Enrolled ∧ c_id,type ∈ Courses ∧ type = \"CS\"}" → "IDs of students enrolled in CS courses"

Return ONLY the one-sentence summary, nothing else."""


async def summarize_relation(
    drc_pretty: str,
    columns: list[str],
    config: LLMClientConfig | None = None,
) -> str:
    """Ask the LLM to summarize a DRC expression in one sentence.

    Args:
        drc_pretty: The pretty-printed DRC expression.
        columns: The output columns of the relation.
        config: LLM client configuration.

    Returns:
        A one-sentence summary string.
    """
    if config is None:
        config = LLMClientConfig()

    client = _get_bedrock_client(config)

    user_message = f"Columns: {columns}\nDRC: {drc_pretty}"

    request_body = {
        "modelId": config.model_id,
        "system": [{"text": _SYSTEM_PROMPT_SUMMARIZE}],
        "messages": [
            {"role": "user", "content": [{"text": user_message}]}
        ],
        "inferenceConfig": {
            "maxTokens": 100,
            "temperature": 0.0,
        },
    }

    response = client.converse(**request_body)

    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    for block in content_blocks:
        if "text" in block:
            return block["text"].strip()

    return f"Relation with columns {columns}"


_SYSTEM_PROMPT_DISTINCT = """You decide whether a natural-language database question requires deduplication in the generated SQL.

Answer YES when the question asks for a *count or list of distinct entities* and the query structure could produce duplicate rows for the same entity. Specifically:

Use DISTINCT = true when:
- "How many X have / are ..." where X is an entity that might appear multiple times in the result due to joins or multiple matching rows (e.g. "how many molecules have a triple bond" — a molecule with 3 triple bonds produces 3 rows; we want distinct molecules)
- "List all X that ..." where X is a single entity that can match the predicate multiple ways (e.g. "employees that have two or more reviews" — joining with reviews duplicates the employee row)
- "Find every / all distinct / unique ..."
- The selected entity column is NOT a primary key of the base table being filtered

Do NOT use DISTINCT when:
- The question asks for one row per occurrence ("list all reviews", "list employees and their review IDs")
- The question is a simple aggregation over a primary key column ("how many orders" where each order row is unique)
- The question pairs multiple entities and each pair is meant to be a distinct row ("list employees with their departments")
- The selected columns already include a unique identifier of every joined entity

IMPORTANT: When DISTINCT is needed and the query uses COUNT, the SQL should emit COUNT(DISTINCT col), not SELECT DISTINCT ... COUNT(col). The use_distinct flag applies to both SELECT DISTINCT (for non-aggregate queries) and COUNT(DISTINCT ...) (for aggregate queries).

When in doubt, prefer not to use DISTINCT — unnecessary DISTINCT can hide bugs."""


_DISTINCT_TOOL = {
    "name": "decide_distinct",
    "description": "Decide whether the SQL query should use SELECT DISTINCT.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Short justification for the decision.",
            },
            "use_distinct": {
                "type": "boolean",
                "description": "True if the SQL should use SELECT DISTINCT, false otherwise.",
            },
        },
        "required": ["reasoning", "use_distinct"],
    },
}


@dataclass
class DistinctDecision:
    """Whether the user's question implies SELECT DISTINCT, with reasoning."""

    use_distinct: bool
    reasoning: str = ""


async def decide_distinct(
    question: str,
    schema: str,
    config: LLMClientConfig | None = None,
) -> DistinctDecision:
    """Ask the LLM whether the user's question implies SELECT DISTINCT.

    The decision is based on the natural-language question and schema only —
    the operation tree is not consulted, because user intent (set vs bag)
    is a question-level concern, not a tree-structure concern.

    Args:
        question: The natural language question.
        schema: The database schema (CREATE TABLE statements).
        config: LLM client configuration.

    Returns:
        A DistinctDecision with the boolean and a short reasoning string.
        Defaults to ``use_distinct=False`` on any error or unexpected
        response, since unnecessary DISTINCT can hide bugs.
    """
    if config is None:
        config = LLMClientConfig()

    try:
        client = _get_bedrock_client(config)
    except ImportError:
        return DistinctDecision(use_distinct=False, reasoning="boto3 unavailable")

    user_message = f"Schema:\n{schema}\n\nQuestion: {question}"

    request_body = {
        "modelId": config.model_id,
        "system": [{"text": _SYSTEM_PROMPT_DISTINCT}],
        "messages": [
            {"role": "user", "content": [{"text": user_message}]}
        ],
        "inferenceConfig": {
            "maxTokens": 256,
            "temperature": 0.0,
        },
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": _DISTINCT_TOOL["name"],
                        "description": _DISTINCT_TOOL["description"],
                        "inputSchema": {"json": _DISTINCT_TOOL["input_schema"]},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "decide_distinct"}},
        },
    }

    try:
        response = client.converse(**request_body)
    except Exception as e:
        return DistinctDecision(use_distinct=False, reasoning=f"LLM error: {e}")

    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    for block in content_blocks:
        if block.get("toolUse") and block["toolUse"].get("name") == "decide_distinct":
            args = block["toolUse"]["input"]
            return DistinctDecision(
                use_distinct=bool(args.get("use_distinct", False)),
                reasoning=str(args.get("reasoning", "")),
            )

    return DistinctDecision(use_distinct=False, reasoning="No tool use in response")
