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
- (exists (vars...) RelationName body) - existential quantifier
- (forall (vars...) RelationName body) - universal quantifier

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
        relations_desc += f"  [{i}] Table '{tr['table_name']}' columns={tr['columns']}\n"
        relations_desc += f"      DRC: {tr['lisp_syntax']}\n"

    for ir in intermediate_relations:
        idx = ir.index
        ir_lisp = ""
        from text_to_sql_planner.printer import print_lisp as _print_lisp, PrintSuccess as _PS
        lisp_result = _print_lisp(ir.expression)
        if isinstance(lisp_result, _PS):
            ir_lisp = lisp_result.output
        relations_desc += f"  [{idx}] Intermediate columns={ir.columns}\n"
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
- Membership: (in (var1 var2 ...) TableName)
- Logical: (and cond1 cond2), (or cond1 cond2), (not cond), (implies cond1 cond2)
- Comparison: (= x y), (!= x y), (< x y), (> x y), (<= x y), (>= x y)
- Quantifiers: (exists (var1 var2 ...) TableName body), (forall (var1 var2 ...) TableName body)
- Literals: strings in double quotes "hello", numbers as-is 42
- Variables: plain identifiers like name, age, id

Example:
Question: "Find all employees in department 5"
Schema: CREATE TABLE employees (id INT, name VARCHAR, dept_id INT)
Answer: (drc (id name dept_id) (and (in (id name dept_id) employees) (= dept_id 5)))

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
