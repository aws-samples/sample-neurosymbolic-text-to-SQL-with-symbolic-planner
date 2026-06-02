"""Tests for the question-converter retry-hint helper.

The helper inspects the prior raw LLM output and the parser error,
then chooses a hint targeted at the apparent failure mode. Before
this module existed the retry prompt always mentioned "over-deep
nesting in the negation of an 'exactly N' pattern", which was
actively misleading for unrelated failures (e.g. the Compensation
join in ``nohup3.md`` that produced a unary ``(and (in (...)
Compensation))``).

The hint must:

- Surface a unary-and/or hint when the prior output contains
  ``(and X)`` or ``(or X)`` with a single immediate operand.
- Surface an imbalanced-paren hint when ``(`` and ``)`` counts
  differ.
- Surface the exactly-N hint only when the output actually contains
  ``(not (exists`` (the construct that actually goes wrong in deep
  nesting).
- Fall back to a generic re-balancing hint otherwise.
"""

from __future__ import annotations

from text_to_sql_planner.converter.question_converter import _retry_hint_for


def test_hint_unary_and():
    text = (
        "(drc (emp_id name) (exists (a) (and (in (a name) Employees)"
        " (exists (b) (and (in (b a) Compensation))))))"
    )
    hint = _retry_hint_for(text, "ParseError at offset 99: Unexpected token: RPAREN")
    assert "unary" in hint.lower()


def test_hint_unary_or():
    text = "(drc (x) (or (in (x) R)))"
    hint = _retry_hint_for(text, "ParseError")
    assert "unary" in hint.lower()


def test_hint_balanced_and_no_unary_falls_back_or_negation():
    """Balanced parens, no unary, no ``(not (exists`` → generic hint."""
    text = "(drc (x) (in (x) R))"
    hint = _retry_hint_for(text, "some error")
    assert "Re-balance" in hint or "re-emit" in hint.lower()


def test_hint_imbalanced_more_closes():
    text = "(drc (x) (in (x) R)))"  # one extra close
    hint = _retry_hint_for(text, "ParseError")
    assert "unbalanced" in hint.lower()
    assert "more closes" in hint.lower()
    assert "off by 1" in hint.lower()


def test_hint_imbalanced_more_opens():
    text = "((((drc (x) (in (x) R))"  # extra opens
    hint = _retry_hint_for(text, "ParseError")
    assert "unbalanced" in hint.lower()
    assert "more opens" in hint.lower()


def test_hint_exactly_n_only_when_relevant():
    """``(not (exists ...))`` in the output → the exactly-N hint kicks in.

    The output is balanced and has no unary and/or, so the previous
    heuristics don't fire. The negation-of-existential pattern is
    what triggers the exactly-N message.
    """
    text = (
        "(drc (x) (and (in (x) R)"
        " (not (exists (r1) (and (in (r1 x) S)"
        " (not (exists (r2) (and (in (r2 x) S) (!= r2 r1)))))))))"
    )
    hint = _retry_hint_for(text, "ParseError")
    assert "exactly N" in hint or '"exactly N"' in hint


def test_hint_no_exactly_n_when_no_negated_exists():
    """An expression without ``(not (exists`` should NOT mention
    exactly-N — that hint was the actively misleading one in
    nohup3.md.
    """
    text = (
        "(drc (emp_id first_name last_name base_salary)"
        " (exists (email) (and (in (emp_id first_name last_name email) Employees)"
        " (exists (comp_id) (in (comp_id emp_id base_salary) Compensation)))))"
    )
    hint = _retry_hint_for(text, "ParseError")
    assert "exactly N" not in hint
    assert '"exactly N"' not in hint


def test_hint_handles_empty_input():
    """Empty / None-like input shouldn't crash; falls through to
    the generic hint."""
    hint = _retry_hint_for("", "no output")
    assert isinstance(hint, str)
    assert hint  # non-empty
