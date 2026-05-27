"""Error types for the Text-to-SQL Planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    """Enumeration of all error codes used across the system."""

    # Input validation
    EMPTY_QUESTION = "EMPTY_QUESTION"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    NO_TABLES_FOUND = "NO_TABLES_FOUND"
    # Parser errors
    PARSE_ERROR = "PARSE_ERROR"
    MAX_DEPTH_EXCEEDED = "MAX_DEPTH_EXCEEDED"
    # Printer errors
    PRINT_ERROR = "PRINT_ERROR"
    INVALID_AST = "INVALID_AST"
    # Conversion errors
    QUESTION_CONVERSION_FAILED = "QUESTION_CONVERSION_FAILED"
    TABLE_CONVERSION_FAILED = "TABLE_CONVERSION_FAILED"
    # Operator errors
    INVALID_OPERATOR_PARAMS = "INVALID_OPERATOR_PARAMS"
    INVALID_INPUT_RELATION = "INVALID_INPUT_RELATION"
    # Equivalence errors
    CVC5_TIMEOUT = "CVC5_TIMEOUT"
    CVC5_PROCESS_ERROR = "CVC5_PROCESS_ERROR"
    CVC5_PARSE_ERROR = "CVC5_PARSE_ERROR"
    # Planner errors
    MAX_ITERATIONS_EXCEEDED = "MAX_ITERATIONS_EXCEEDED"
    OPERATOR_SELECTION_FAILED = "OPERATOR_SELECTION_FAILED"
    # SQL conversion errors
    SQL_CONVERSION_FAILED = "SQL_CONVERSION_FAILED"
    INCOMPLETE_TREE = "INCOMPLETE_TREE"
    # LLM errors
    LLM_API_ERROR = "LLM_API_ERROR"
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"


@dataclass
class TextToSQLError(Exception):
    """Top-level error type, raisable as an exception.

    Carries an :class:`ErrorCode`, a human-readable message, and an optional
    context dictionary for additional debugging information.
    """

    code: ErrorCode
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"


@dataclass
class ParseError(Exception):
    """Returned by the parser when input cannot be parsed.

    Includes the character offset where parsing failed, a descriptive message,
    and optionally the surrounding characters for debugging.

    Inherits from :class:`Exception` so the lexer can ``raise`` it directly
    while still keeping the structural dataclass interface used by the
    parser's ``ParserFailure`` result type.

    Validates: Requirements 4.2
    """

    offset: int
    message: str
    context_str: str = ""  # surrounding characters for debugging

    def __str__(self) -> str:
        ctx = f" (near {self.context_str!r})" if self.context_str else ""
        return f"ParseError at offset {self.offset}: {self.message}{ctx}"


@dataclass
class PrintError:
    """Returned by printers when an AST cannot be serialized.

    Carries a descriptive message and optionally the offending node.

    Validates: Requirements 5.5, 6.4
    """

    message: str
    node: Optional[Any] = None  # the node that caused the failure
