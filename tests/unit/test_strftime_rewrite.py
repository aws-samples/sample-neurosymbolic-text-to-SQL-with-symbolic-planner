"""Tests for the STRFTIME year-comparison rewrite.

Background — run-12 dev_947:
BIRD's gold SQL was

    SELECT COUNT(driverId) FROM drivers
    WHERE nationality = 'British' AND STRFTIME('%Y', dob) > '1980'

while the planner generated

    SELECT COUNT(driverId) FROM drivers
    WHERE nationality = 'British' AND dob > '1980-12-31'

These describe the same set of rows over real dates (every dob with
year > 1980 is also dob > 1980-12-31, and vice versa), but cvc5 saw
``STRFTIME`` as an uninterpreted function and reported
``not_equivalent``. The execution check matched on actual rows, but
the logical verdict was a false negative.

The fix: rewrite ``STRFTIME('%Y', d) <op> 'YYYY'`` patterns into
integer comparisons on ``d`` (days since 1970-01-01) before the
SMT script is built. After the rewrite both sides become arithmetic
constraints on the date column and cvc5 proves equivalence cleanly.
"""

from __future__ import annotations

import pytest

from text_to_sql_planner.equivalence.smt_preprocessing import (
    rewrite_strftime_year_comparisons,
)
from text_to_sql_planner.types.drc import (
    ComparisonNode,
    FunctionCallNode,
    LiteralNode,
    LogicalConnectiveNode,
    MembershipNode,
    NotNode,
    QuantifierNode,
    VariableRefNode,
)


def _strftime_year(date_var: str) -> FunctionCallNode:
    """``STRFTIME('%Y', <date_var>)`` constructor."""
    return FunctionCallNode(
        function="STRFTIME",
        arguments=[
            LiteralNode(value="%Y", data_type="string"),
            VariableRefNode(name=date_var),
        ],
    )


def _year_lit(year: int) -> LiteralNode:
    """Year literal as a string (matches BIRD's gold-SQL form)."""
    return LiteralNode(value=str(year), data_type="string")


# ---------------------------------------------------------------------------
# Operator-by-operator coverage
#
# Each test asserts that the rewrite produces an integer comparison
# whose right-hand side is the correct days-since-1970 boundary.
# ---------------------------------------------------------------------------


def test_rewrite_greater_than_uses_year_plus_one_jan1():
    """``STRFTIME('%Y', d) > '1980'`` ↔ ``d >= 4018`` (Jan 1, 1981)."""
    cond = ComparisonNode(
        operator=">",
        left=_strftime_year("dob"),
        right=_year_lit(1980),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, ComparisonNode)
    assert out.operator == ">="
    assert isinstance(out.left, VariableRefNode)
    assert out.left.name == "dob"
    assert isinstance(out.right, LiteralNode)
    # Jan 1, 1981 is 11*365 + 3 = 4018 days after 1970-01-01.
    assert out.right.value == 4018


def test_rewrite_greater_or_equal_uses_year_jan1():
    """``STRFTIME('%Y', d) >= '1980'`` ↔ ``d >= 3652`` (Jan 1, 1980)."""
    cond = ComparisonNode(
        operator=">=",
        left=_strftime_year("dob"),
        right=_year_lit(1980),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, ComparisonNode)
    assert out.operator == ">="
    assert out.right.value == 3652  # 1980-01-01


def test_rewrite_less_than_uses_year_jan1():
    """``STRFTIME('%Y', d) < '1980'`` ↔ ``d < 3652``."""
    cond = ComparisonNode(
        operator="<",
        left=_strftime_year("dob"),
        right=_year_lit(1980),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, ComparisonNode)
    assert out.operator == "<"
    assert out.right.value == 3652


def test_rewrite_less_or_equal_uses_year_plus_one_jan1():
    """``STRFTIME('%Y', d) <= '1980'`` ↔ ``d < 4018``."""
    cond = ComparisonNode(
        operator="<=",
        left=_strftime_year("dob"),
        right=_year_lit(1980),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, ComparisonNode)
    assert out.operator == "<"
    assert out.right.value == 4018


def test_rewrite_equality_becomes_range_and():
    """``STRFTIME('%Y', d) = '1980'`` ↔ ``3652 <= d < 4018`` (Jan 1
    of 1980 inclusive, Jan 1 of 1981 exclusive)."""
    cond = ComparisonNode(
        operator="=",
        left=_strftime_year("dob"),
        right=_year_lit(1980),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, LogicalConnectiveNode)
    assert out.operator == "and"
    # Lower bound on the left.
    assert isinstance(out.left, ComparisonNode)
    assert out.left.operator == ">="
    assert out.left.right.value == 3652
    # Upper bound on the right.
    assert isinstance(out.right, ComparisonNode)
    assert out.right.operator == "<"
    assert out.right.right.value == 4018


def test_rewrite_inequality_becomes_range_or():
    """``STRFTIME('%Y', d) != '1980'`` ↔ ``d < 3652 OR d >= 4018``."""
    cond = ComparisonNode(
        operator="!=",
        left=_strftime_year("dob"),
        right=_year_lit(1980),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, LogicalConnectiveNode)
    assert out.operator == "or"
    assert out.left.operator == "<"
    assert out.left.right.value == 3652
    assert out.right.operator == ">="
    assert out.right.right.value == 4018


# ---------------------------------------------------------------------------
# Operand-flipping: literal-on-the-left form must be handled too
# ---------------------------------------------------------------------------


def test_rewrite_handles_flipped_operands():
    """``'1980' < STRFTIME('%Y', d)`` ↔ ``d >= 4018`` (the same as
    ``STRFTIME(...) > '1980'`` after flipping the operator)."""
    cond = ComparisonNode(
        operator="<",
        left=_year_lit(1980),
        right=_strftime_year("dob"),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, ComparisonNode)
    assert out.operator == ">="
    assert out.right.value == 4018


# ---------------------------------------------------------------------------
# Non-rewritable patterns pass through unchanged
# ---------------------------------------------------------------------------


def test_rewrite_skips_non_year_format():
    """``STRFTIME('%m', d) > '06'`` is NOT rewritten (only ``%Y``
    has a clean integer-date encoding in this pass)."""
    cond = ComparisonNode(
        operator=">",
        left=FunctionCallNode(
            function="STRFTIME",
            arguments=[
                LiteralNode(value="%m", data_type="string"),
                VariableRefNode(name="dob"),
            ],
        ),
        right=LiteralNode(value="06", data_type="string"),
    )
    out = rewrite_strftime_year_comparisons(cond)
    # Must round-trip — STRFTIME call still present.
    assert isinstance(out, ComparisonNode)
    assert isinstance(out.left, FunctionCallNode)
    assert out.left.function == "STRFTIME"


def test_rewrite_skips_non_literal_year():
    """``STRFTIME('%Y', d) > some_var`` is NOT rewritten — we only
    handle the literal-year case."""
    cond = ComparisonNode(
        operator=">",
        left=_strftime_year("dob"),
        right=VariableRefNode(name="threshold"),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, ComparisonNode)
    assert isinstance(out.left, FunctionCallNode)


def test_rewrite_skips_non_four_digit_year():
    """``STRFTIME('%Y', d) > 'abc'`` (or ``'19'`` or ``'19800'``) is
    NOT rewritten — the year-literal parser rejects malformed input."""
    cond = ComparisonNode(
        operator=">",
        left=_strftime_year("dob"),
        right=LiteralNode(value="abc", data_type="string"),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, ComparisonNode)
    assert isinstance(out.left, FunctionCallNode)


def test_rewrite_leaves_non_strftime_comparisons_alone():
    """A plain ``a > b`` comparison passes through untouched (no
    spurious rewrites of unrelated comparisons)."""
    cond = ComparisonNode(
        operator=">",
        left=VariableRefNode(name="age"),
        right=LiteralNode(value=18, data_type="number"),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, ComparisonNode)
    assert out.operator == ">"
    assert out.left.name == "age"
    assert out.right.value == 18


# ---------------------------------------------------------------------------
# Recursion through compound forms
# ---------------------------------------------------------------------------


def test_rewrite_descends_into_and():
    """The rewrite reaches STRFTIME calls nested inside an AND chain."""
    cond = LogicalConnectiveNode(
        operator="and",
        left=ComparisonNode(
            operator="=",
            left=VariableRefNode(name="nationality"),
            right=LiteralNode(value="British", data_type="string"),
        ),
        right=ComparisonNode(
            operator=">",
            left=_strftime_year("dob"),
            right=_year_lit(1980),
        ),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, LogicalConnectiveNode)
    # Left (nationality equality) unchanged.
    assert isinstance(out.left, ComparisonNode)
    assert out.left.operator == "="
    # Right (STRFTIME) rewritten to integer comparison.
    assert isinstance(out.right, ComparisonNode)
    assert out.right.operator == ">="
    assert out.right.right.value == 4018


def test_rewrite_descends_into_quantifier_body():
    """STRFTIME inside an ``∃`` body still gets rewritten."""
    cond = QuantifierNode(
        kind="exists",
        variables=["dob"],
        body=ComparisonNode(
            operator=">",
            left=_strftime_year("dob"),
            right=_year_lit(2000),
        ),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, QuantifierNode)
    assert isinstance(out.body, ComparisonNode)
    assert out.body.operator == ">="


def test_rewrite_descends_into_negation():
    """STRFTIME inside ``NOT (…)`` still gets rewritten."""
    cond = NotNode(
        operand=ComparisonNode(
            operator=">",
            left=_strftime_year("dob"),
            right=_year_lit(1980),
        ),
    )
    out = rewrite_strftime_year_comparisons(cond)
    assert isinstance(out, NotNode)
    assert isinstance(out.operand, ComparisonNode)
    assert out.operand.operator == ">="


# ---------------------------------------------------------------------------
# End-to-end against real cvc5: dev_947 false-negative resolves.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dev_947_strftime_vs_date_comparison_resolves_with_real_cvc5():
    """Pre-fix this returned ``not_equivalent`` because cvc5 couldn't
    relate the two formulations. With the rewrite both sides reduce
    to integer comparisons on ``dob`` and cvc5 proves them equivalent.
    """
    import os
    import shutil

    cvc5 = shutil.which("cvc5") or "/usr/local/bin/cvc5"
    if not os.path.exists(cvc5):
        pytest.skip("cvc5 binary not available")

    from text_to_sql_planner.equivalence import (
        EquivalenceCheckerConfig,
        EquivalentResult,
        check_equivalence,
    )
    from text_to_sql_planner.types.drc import (
        AggregateVariable,
        DRCExpression,
    )

    # Generated: COUNT(driverId) WHERE nationality="British" AND dob>"1980-12-31"
    gen = DRCExpression(
        result_variables=[AggregateVariable(function="COUNT", column="driverId")],
        condition=QuantifierNode(
            kind="exists",
            variables=["nationality", "dob"],
            body=LogicalConnectiveNode(
                operator="and",
                left=LogicalConnectiveNode(
                    operator="and",
                    left=MembershipNode(
                        variables=["driverId", "nationality", "dob"],
                        relation="drivers",
                    ),
                    right=ComparisonNode(
                        operator="=",
                        left=VariableRefNode(name="nationality"),
                        right=LiteralNode(value="British", data_type="string"),
                    ),
                ),
                right=ComparisonNode(
                    operator=">",
                    left=VariableRefNode(name="dob"),
                    right=LiteralNode(value="1980-12-31", data_type="string"),
                ),
            ),
        ),
    )

    # Gold: COUNT(driverId) WHERE nationality="British" AND STRFTIME("%Y",dob)>"1980"
    gold = DRCExpression(
        result_variables=[AggregateVariable(function="COUNT", column="driverId")],
        condition=QuantifierNode(
            kind="exists",
            variables=["nationality", "dob"],
            body=LogicalConnectiveNode(
                operator="and",
                left=LogicalConnectiveNode(
                    operator="and",
                    left=MembershipNode(
                        variables=["driverId", "nationality", "dob"],
                        relation="drivers",
                    ),
                    right=ComparisonNode(
                        operator="=",
                        left=VariableRefNode(name="nationality"),
                        right=LiteralNode(value="British", data_type="string"),
                    ),
                ),
                right=ComparisonNode(
                    operator=">",
                    left=_strftime_year("dob"),
                    right=_year_lit(1980),
                ),
            ),
        ),
    )

    config = EquivalenceCheckerConfig(cvc5_path=cvc5, timeout_seconds=20.0)
    result = await check_equivalence(gen, gold, config=config)
    assert isinstance(result, EquivalentResult), (
        f"expected equivalent, got {type(result).__name__}: {result}"
    )
