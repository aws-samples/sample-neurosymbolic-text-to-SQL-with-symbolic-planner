# Text-to-SQL Planner

A Text-to-SQL agent that converts natural language questions into provably correct SQL queries using a symbolic planner grounded in Domain Relational Calculus (DRC) and the cvc5 theorem prover.

## How it works

1. You provide a natural language question and a database schema (CREATE TABLE statements)
2. Claude (via AWS Bedrock) converts the question into a formal DRC expression
3. The planner iteratively applies relational algebra operators to build toward the target
4. At each step, cvc5 verifies logical equivalence between the current result and the target
5. Once equivalence is proven, the operation tree is translated into a SQL SELECT statement

## Requirements

- Ubuntu 26.04 LTS (or compatible Linux distribution)
- Python 3.11+
- AWS credentials configured (IAM role, environment variables, or `~/.aws/credentials`)
- Access to Claude on Amazon Bedrock (model: `global.anthropic.claude-opus-4-6-v1`)
- cvc5 binary on PATH (for equivalence checking) — see install instructions below

## Installation

### Install cvc5

Download the static binary for your platform from the [cvc5 releases page](https://github.com/cvc5/cvc5/releases):

```bash
# Linux arm64 (aarch64)
curl -L -o /tmp/cvc5.zip https://github.com/cvc5/cvc5/releases/download/cvc5-1.3.4/cvc5-Linux-arm64-static.zip

# Linux x86_64
# curl -L -o /tmp/cvc5.zip https://github.com/cvc5/cvc5/releases/download/cvc5-1.3.4/cvc5-Linux-x86_64-static.zip

unzip -o /tmp/cvc5.zip -d /tmp/cvc5-extract
sudo cp /tmp/cvc5-extract/cvc5-Linux-*/bin/cvc5 /usr/local/bin/cvc5
sudo chmod +x /usr/local/bin/cvc5

# Verify
cvc5 --version
```

### Install the Python project

```bash
cd /home/ubuntu
uv sync
```

This installs:
- `boto3` — AWS SDK for Bedrock LLM calls
- `pydantic` — data validation
- `hypothesis` — property-based testing (dev)
- `pytest` / `pytest-asyncio` — test runner (dev)

## Usage

### Command line

After installation, run via `uv`:

```bash
# Basic usage with inline question and schema
uv run python -m text_to_sql_planner \
    -q "Which students have taken Computer Science courses?" \
    -s "CREATE TABLE Students (s_id INT, name VARCHAR);
        CREATE TABLE Enrolled (s_id INT, c_id INT);
        CREATE TABLE Courses (c_id INT, c_type VARCHAR);"

# Read schema from a file
uv run python -m text_to_sql_planner -q "Find all employees in department 5" -f schema.sql

# Pipe the question via stdin
echo "How many orders were placed last month?" | uv run python -m text_to_sql_planner -f schema.sql

# Verbose mode (shows target DRC expression)
uv run python -m text_to_sql_planner -q "Find all users" -f schema.sql -v

# Custom AWS region and model
uv run python -m text_to_sql_planner -q "Find all users" -f schema.sql --region us-west-2

# Custom cvc5 path and timeout
uv run python -m text_to_sql_planner -q "Find all users" -f schema.sql --cvc5-path /usr/local/bin/cvc5 --cvc5-timeout 60
```

The command prints the SQL to stdout and errors to stderr. Exit code 0 on success, 1 on failure.

### Quick start (canned example)

Run the included example that uses a university schema:

```bash
uv run python example.py
```

This asks "Which students have taken one or more Computer Science courses?" against a Students/Courses/Enrolled schema and prints the generated SQL. You can also use the CLI with the included schema file:

```bash
uv run python -m text_to_sql_planner \
    -q "How many students got a grade above 90?" \
    -f example_schema.sql
```

### Python API

### Python API

```python
import asyncio
from text_to_sql_planner import run

async def main():
    question = "Which students have taken one or more Computer Science courses?"
    schema = """
    CREATE TABLE Students (s_id INT, name VARCHAR);
    CREATE TABLE Enrolled (s_id INT, c_id INT);
    CREATE TABLE Courses (c_id INT, c_type VARCHAR);
    """

    result = await run(question=question, schema=schema)

    from text_to_sql_planner.main import TextToSQLSuccess, TextToSQLFailure

    if isinstance(result, TextToSQLSuccess):
        print("SQL:", result.sql)
    else:
        print("Error:", result.error)

asyncio.run(main())
```

Save as `example.py` and run:

```bash
python3 example.py
```

### Configuration

You can customize the region, model, planner limits, and equivalence checker timeout:

```python
from text_to_sql_planner.planner import PlannerConfig
from text_to_sql_planner.planner.llm_client import LLMClientConfig
from text_to_sql_planner.equivalence import EquivalenceCheckerConfig

config = PlannerConfig(
    max_iterations=50,
    max_retries_per_iteration=5,
    initial_temperature=0.0,
    temperature_step=0.2,
    llm_config=LLMClientConfig(
        region="us-east-1",
        model_id="global.anthropic.claude-opus-4-6-v1",
        max_tokens=4096,
    ),
    equivalence_config=EquivalenceCheckerConfig(
        cvc5_path="cvc5",
        timeout_seconds=30.0,
    ),
)

result = await run(question=question, schema=schema, config=config)
```

### AWS credentials

The system uses boto3's standard credential resolution. Any of these work:

- IAM instance role (if running on EC2/ECS/Lambda)
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`
- Shared credentials file: `~/.aws/credentials`
- AWS SSO / `aws configure sso`

No Anthropic API key is needed.

## Running the test suite

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run a specific test file
uv run python -m pytest tests/unit/test_parser.py -v

# Run with short traceback on failure
uv run python -m pytest tests/ --tb=short
```

The test suite (275 tests) covers:
- Lexer tokenization
- Parser (recursive descent, nesting depth, error reporting)
- Lisp printer (serialization, all-or-nothing semantics)
- Pretty printer (Unicode formatting, precedence)
- Table converter (CREATE TABLE parsing)
- All 6 RA operators (selection, join, projection, cartesian product, union, division)
- SMT-LIB conversion
- Equivalence checker (arity short-circuit, mock subprocess scenarios)
- SQL converter (all operator mappings, nested operations, error cases)
- Main entry point (input validation, orchestration with mocked LLM)

Tests that exercise the LLM or cvc5 use mocks — no external services are needed to run the test suite.

## Project structure

```
text_to_sql_planner/
├── __init__.py              # Package entry point (exports `run`)
├── main.py                  # Pipeline orchestrator
├── types/
│   ├── drc.py               # DRC AST node types
│   ├── operators.py         # RA operator types
│   ├── operation_tree.py    # Operation tree types
│   └── errors.py            # Error types
├── parser/
│   ├── lexer.py             # Lisp S-expression tokenizer
│   └── parser.py            # Recursive descent parser
├── printer/
│   ├── lisp_printer.py      # AST → Lisp S-expression
│   └── pretty_printer.py    # AST → Unicode notation (∀, ∃, ∧, ∨, ¬, ∈, →)
├── converter/
│   ├── table_converter.py   # CREATE TABLE → DRC
│   └── question_converter.py # NL question → DRC (via LLM)
├── operators/
│   ├── selection.py         # σ (filter)
│   ├── join.py              # ⋈ (natural join)
│   ├── projection.py        # π (column subset)
│   ├── cartesian_product.py # × (cross product)
│   ├── union_op.py          # ∪ (set union)
│   └── division.py          # ÷ (relational division)
├── equivalence/
│   ├── smt_converter.py     # DRC → SMT-LIB syntax
│   └── equivalence_checker.py # Parallel cvc5 invocation
├── planner/
│   ├── llm_client.py        # boto3 Bedrock wrapper for Claude
│   └── planner.py           # Iterative planning loop
└── sql/
    └── sql_converter.py     # Operation tree → SQL SELECT
```
