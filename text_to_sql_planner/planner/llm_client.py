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
- join: Natural join on shared columns. Requires 2 inputs.
- cartesian_product: Cross product of two relations. Requires 2 inputs.
- union: Set union of two compatible relations. Requires 2 inputs.
- division: Relational division. Requires 2 inputs.

DRC Notation Guide:
- Expressions use Lisp S-expression syntax
- (drc (result-vars) condition) - top-level expression
- (in (vars...) RelationName) - membership
- (and cond1 cond2), (or cond1 cond2), (not cond) - logical
- (= x y), (!= x y), (< x y), (> x y), (<= x y), (>= x y) - comparison
- (exists (vars...) body) - existential quantifier (use (in ...) in body to bind to a relation)
- (forall (vars...) body) - universal quantifier (use (in ...) in body to bind to a relation)

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
                    "cartesian_product",
                    "union",
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
                "description": "Operator-specific parameters. For selection: {condition: <lisp-string>}. For projection: {columns: [col1, col2]}. For join: {join_columns: [col]}. For cartesian_product/union/division: {}.",
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
  - Aggregate: (COUNT col), (SUM col), (AVG col), (MIN col), (MAX col)
- Membership: (in (var1 var2 ...) TableName) — asserts that the tuple (var1, var2, ...) is a row in TableName
- Logical: (and cond1 cond2), (or cond1 cond2), (not cond), (implies cond1 cond2)
- Comparison: (= x y), (!= x y), (< x y), (> x y), (<= x y), (>= x y)
- Quantifiers: (exists (var1 var2 ...) body), (forall (var1 var2 ...) body)
  - Quantifiers just bind variables. Use (in ...) inside the body to constrain them to a relation.
- Date functions: CURRENT_DATE (today's date), (DATE_SUB expr days) (subtract days from a date), (DATEDIFF expr1 expr2) (days between two dates)
- Literals: strings in double quotes "hello", numbers as-is 42
- Variables: plain identifiers like name, age, id

IMPORTANT RULES:
- Result variables are FREE variables — they must NOT appear as quantified variables.
- CRITICAL: Every variable in a membership (in ...) that is NOT a result variable and NOT the column being aggregated MUST be wrapped in an existential quantifier (exists ...). There must be NO free variables in the condition other than the result variables (and aggregate columns). If a table has 10 columns but you only need 2 as result variables, the other 8 MUST be existentially quantified.
- When the question asks "how many", "count", "total number of", etc., use (COUNT col) in the result variables.
- For age/date calculations, use CURRENT_DATE and (DATE_SUB CURRENT_DATE days). For example, "at least 30 years old" means (>= (DATE_SUB CURRENT_DATE 10950) date_of_birth) where 10950 = 30*365.
- Quantifiers bind variables; membership (in) constrains them to a table. Always pair them.

Examples:

Question: "Find all employees in department 5"
Schema: CREATE TABLE employees (id INT, name VARCHAR, dept_id INT)
Answer: (drc (id name dept_id) (and (in (id name dept_id) employees) (= dept_id 5)))

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

Cardinality patterns ("at least N", "two or more", "more than N", "exactly N"):
- "at least N" / "N or more" / "two or more" → assert the existence of N distinct witnesses, joined by AND, all pairwise distinct. DO NOT add ``(not (exists ...))`` for an extra witness — that turns "≥N" into "exactly N" or "≥N+1".
- "more than N" → same as "at least N+1".
- "exactly N" → assert N distinct witnesses AND ``(not (exists ...))`` for an (N+1)-th witness distinct from all the prior ones.
- Each witness must be wrapped in its own existential quantifier and constrained by an (in ...) membership.
- Witnesses must be pairwise distinct via (!= ...) on the identifying columns.

Question: "List all employees that have two or more performance reviews"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR); CREATE TABLE Reviews (review_id INT, emp_id INT, rating INT)
Answer: (drc (emp_id name) (and (in (emp_id name) Employees) (exists (r1 rating1) (and (in (r1 emp_id rating1) Reviews) (exists (r2 rating2) (and (in (r2 emp_id rating2) Reviews) (!= r1 r2)))))))
Note: TWO existential witnesses (r1 and r2), both bound to ``emp_id``, with ``(!= r1 r2)``. NO third witness, NO ``(not (exists ...))`` clause.

Question: "List employees with exactly two performance reviews"
Schema: CREATE TABLE Employees (emp_id INT, name VARCHAR); CREATE TABLE Reviews (review_id INT, emp_id INT, rating INT)
Answer: (drc (emp_id name) (and (in (emp_id name) Employees) (and (exists (r1 rating1) (and (in (r1 emp_id rating1) Reviews) (exists (r2 rating2) (and (in (r2 emp_id rating2) Reviews) (!= r1 r2))))) (not (exists (r3 rating3) (and (in (r3 emp_id rating3) Reviews) (and (!= r3 r1) (!= r3 r2))))))))
Note: TWO positive witnesses AND a (not (exists ...)) for a third one. The "exactly" is what introduces the negative clause.

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
Answer: (limit 3 (order-by (((COUNT emp_id) desc)) (drc (dept_name (COUNT emp_id)) (exists (dept_id) (and (in (dept_id dept_name) Departments) (exists (emp_id name) (and (in (emp_id name dept_id) Employees))))))))
Note: The order-by key matches the aggregate result variable. The aggregate appears in BOTH the result variables of the inner DRC AND as the order-by key.

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


_SYSTEM_PROMPT_DISTINCT = """You decide whether a natural-language database question requires SELECT DISTINCT in the generated SQL.

Use DISTINCT when the question asks for a *set* of distinct entities and the join structure could otherwise produce duplicate rows. Common signals:
- "list all X that ..." or "which X ..." where X is a single entity that can match the predicate multiple ways (e.g. "employees that have two or more reviews" — joining with reviews twice would duplicate the employee otherwise)
- "find every / all distinct / unique ..."
- The selected columns don't include a unique identifier of the joined-in entities

Do NOT use DISTINCT when:
- The question asks for one row per occurrence ("list all reviews", "list employees and their review IDs")
- The question is an aggregation ("how many X", "average Y", "sum Z") — aggregates handle distinctness themselves
- The question pairs multiple entities and each pair is meant to be a distinct row ("list employees with their departments")
- The selected columns already include a unique identifier of every joined entity

When in doubt, prefer not to use DISTINCT, since aggregations like COUNT count duplicates by design and unnecessary DISTINCT can hide bugs."""


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
