"""Tests for the planner's negative-memory mechanism.

When the planner rejects an LLM-proposed operator (because of validation,
duplicate output, or a not-equivalent verdict), it surfaces that
proposal back to the LLM on the next retry as a ``RejectedProposal`` so
the model doesn't loop on the same pick.

These tests exercise both layers of the wiring:

- :func:`text_to_sql_planner.planner.llm_client.select_operator` — does
  the rejection list end up in the user message it sends to Bedrock?
- :func:`text_to_sql_planner.planner.planner.plan` — does the loop
  accumulate rejections across retries within an iteration, and reset
  between iterations?
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

import text_to_sql_planner.planner.llm_client as llm_client
import text_to_sql_planner.planner.planner as planner_mod
from text_to_sql_planner.converter.table_converter import TableRelation
from text_to_sql_planner.equivalence import (
    EquivalentResult,
    IndeterminateResult,
    NotEquivalentResult,
)
from text_to_sql_planner.planner.llm_client import (
    OperatorSelection,
    RejectedProposal,
    select_operator,
)
from text_to_sql_planner.planner.planner import PlannerConfig, PlannerError, plan
from text_to_sql_planner.types.drc import (
    ColumnVariable,
    DRCExpression,
    MembershipNode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_relation(columns: list[str], relation_name: str) -> DRCExpression:
    return DRCExpression(
        result_variables=[ColumnVariable(name=c) for c in columns],
        condition=MembershipNode(variables=list(columns), relation=relation_name),
    )


def _make_table_relation(columns: list[str], name: str) -> TableRelation:
    return TableRelation(
        table_name=name,
        columns=columns,
        expression=_make_relation(columns, name),
    )


@dataclass
class _CapturedConverse:
    """Records the user message sent to Bedrock on each call."""

    messages: list[str]


def _stub_converse(
    monkeypatch: pytest.MonkeyPatch,
    selections: list[OperatorSelection],
) -> _CapturedConverse:
    """Replace the Bedrock client so each ``client.converse`` returns the
    next pre-baked tool-use response from ``selections``.

    Captures the user-message text on each call so the test can assert
    against the prompt that was sent.
    """
    captured = _CapturedConverse(messages=[])
    iter_selections = iter(selections)

    class _FakeClient:
        def converse(self, **kwargs):
            # Capture the user prompt for later assertions.
            for msg in kwargs.get("messages", []):
                if msg.get("role") == "user":
                    for block in msg.get("content", []):
                        if "text" in block:
                            captured.messages.append(block["text"])

            try:
                sel = next(iter_selections)
            except StopIteration:
                pytest.fail(
                    "Stub ran out of pre-baked OperatorSelection responses"
                )

            return {
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "name": "select_operator",
                                    "input": {
                                        "operator": sel.operator,
                                        "input_indices": list(sel.input_indices),
                                        "params": dict(sel.params),
                                        "reasoning": sel.reasoning,
                                    },
                                }
                            }
                        ]
                    }
                }
            }

    def _fake_get_client(_config):
        return _FakeClient()

    monkeypatch.setattr(llm_client, "_get_bedrock_client", _fake_get_client)
    return captured


# ---------------------------------------------------------------------------
# select_operator: prompt-shape tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_operator_omits_section_without_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no rejections are passed, the prompt has no REJECTED section."""
    captured = _stub_converse(
        monkeypatch,
        [
            OperatorSelection(
                operator="projection",
                input_indices=[0],
                params={"columns": ["id"]},
                reasoning="trim to id",
            )
        ],
    )

    table_descs = [
        {
            "table_name": "T",
            "columns": ["id", "name"],
            "lisp_syntax": "(drc (id name) (in (id name) T))",
            "summary": "All rows from T",
        }
    ]

    sel = await select_operator(
        table_relations=table_descs,
        intermediate_relations=[],
        target_relation="(drc (id) (exists (name) (in (id name) T)))",
        temperature=0.0,
    )

    assert sel.operator == "projection"
    assert len(captured.messages) == 1
    assert "REJECTED PROPOSALS" not in captured.messages[0]


@pytest.mark.asyncio
async def test_select_operator_surfaces_rejections_in_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty rejection list shows up in the prompt with all the
    fields (operator, input indices, params, reason)."""
    captured = _stub_converse(
        monkeypatch,
        [
            OperatorSelection(
                operator="join",
                input_indices=[0, 1],
                params={"join_columns": ["id"]},
                reasoning="try a join instead",
            )
        ],
    )

    rejected = [
        RejectedProposal(
            operator="projection",
            input_indices=[0],
            params={"columns": ["id"]},
            reason="output is identical to existing relation [2]",
        ),
        RejectedProposal(
            operator="difference",
            input_indices=[3, 4],
            params={},
            reason="not equivalent",
        ),
    ]

    table_descs = [
        {
            "table_name": "T",
            "columns": ["id", "name"],
            "lisp_syntax": "(drc (id name) (in (id name) T))",
            "summary": "All rows from T",
        }
    ]

    await select_operator(
        table_relations=table_descs,
        intermediate_relations=[],
        target_relation="(drc (id) (exists (name) (in (id name) T)))",
        temperature=0.2,
        rejected_proposals=rejected,
    )

    assert len(captured.messages) == 1
    prompt = captured.messages[0]
    assert "REJECTED PROPOSALS" in prompt
    assert "DO NOT repeat" in prompt
    # First rejection
    assert "projection" in prompt
    assert "[0]" in prompt
    assert "identical to existing relation [2]" in prompt
    # Second rejection
    assert "difference" in prompt
    assert "[3, 4]" in prompt
    assert "not equivalent" in prompt


# ---------------------------------------------------------------------------
# plan(): negative memory accumulates across retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_accumulates_rejections_within_an_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicate proposal at retry 0 should appear in the rejection
    list passed to ``select_operator`` at retry 1.

    The scenario: the planner has table T = {id, name}, target is the
    same. Iteration 1 retry 0: LLM proposes a projection that
    *reproduces* T (so duplicate-skipped). Iteration 1 retry 1: LLM
    proposes a different projection (id only) that is genuinely new and
    structurally equivalent to the target. We assert that retry 1's
    `select_operator` was called with a non-empty
    `rejected_proposals` containing the retry-0 pick.
    """
    # Spy on select_operator: capture every call's rejected_proposals.
    captured_rejections: list[list[RejectedProposal]] = []
    real_select = llm_client.select_operator
    call_count = {"n": 0}

    async def _spy_select(**kwargs):
        # Snapshot the rejection list at call time.
        captured_rejections.append(list(kwargs.get("rejected_proposals") or []))
        call_count["n"] += 1
        # Fabricate two responses in sequence:
        #   call 0: a duplicate of T (will be rejected by duplicate guard)
        #   call 1: a non-duplicate projection
        if call_count["n"] == 1:
            return OperatorSelection(
                operator="projection",
                input_indices=[0],
                params={"columns": ["id", "name"]},  # same shape as T → duplicate
                reasoning="trim — but actually identical",
            )
        return OperatorSelection(
            operator="projection",
            input_indices=[0],
            params={"columns": ["id"]},
            reasoning="project to id",
        )

    monkeypatch.setattr(planner_mod, "select_operator", _spy_select)

    # Stub equivalence checking: anything we produce is "equivalent" so
    # the loop terminates after the second proposal.
    async def _eq_always_equivalent(*args, **kwargs):
        return EquivalentResult()

    monkeypatch.setattr(planner_mod, "check_equivalence", _eq_always_equivalent)

    # Stub summarize_relation to avoid LLM calls.
    async def _summary_stub(*args, **kwargs):
        return "fake summary"

    monkeypatch.setattr(planner_mod, "summarize_relation", _summary_stub)

    table = _make_table_relation(["id", "name"], "T")
    # Use a target that won't trigger the degenerate-case fast-path so
    # we always exercise the iteration loop.
    target = _make_relation(["id"], "T")

    config = PlannerConfig(max_iterations=2, max_retries_per_iteration=5)
    result = await plan([table], target, config)

    # The two select_operator calls correspond to the duplicate-rejected
    # retry and the successful retry.
    assert call_count["n"] >= 2
    # Retry 0: no rejections yet.
    assert captured_rejections[0] == []
    # Retry 1: the duplicate pick should be surfaced as a rejection.
    assert len(captured_rejections[1]) == 1
    rp = captured_rejections[1][0]
    assert rp.operator == "projection"
    assert rp.input_indices == [0]
    assert rp.params == {"columns": ["id", "name"]}
    assert "identical" in rp.reason.lower()


@pytest.mark.asyncio
async def test_plan_resets_rejections_between_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a retry succeeds (relation added), the next iteration starts
    with a fresh empty rejection list — old rejections from the prior
    iteration don't carry forward."""
    captured_rejections: list[list[RejectedProposal]] = []
    call_count = {"n": 0}

    async def _spy_select(**kwargs):
        captured_rejections.append(list(kwargs.get("rejected_proposals") or []))
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Iteration 1 retry 0: identity projection of T → duplicate.
            return OperatorSelection(
                operator="projection",
                input_indices=[0],
                params={"columns": ["id", "name", "age"]},
                reasoning="dup",
            )
        if call_count["n"] == 2:
            # Iteration 1 retry 1: project to (id, name) — genuinely new,
            # but the equivalence check will say "not equivalent" so we
            # advance to iteration 2.
            return OperatorSelection(
                operator="projection",
                input_indices=[0],
                params={"columns": ["id", "name"]},
                reasoning="step toward target",
            )
        # Iteration 2 retry 0: project to (id, age) — genuinely new, and
        # the equivalence check will succeed.
        return OperatorSelection(
            operator="projection",
            input_indices=[0],
            params={"columns": ["id", "age"]},
            reasoning="rederive",
        )

    monkeypatch.setattr(planner_mod, "select_operator", _spy_select)

    eq_calls = {"n": 0}

    async def _eq_then_eq(*args, **kwargs):
        eq_calls["n"] += 1
        if eq_calls["n"] == 1:
            return NotEquivalentResult()
        return EquivalentResult()

    monkeypatch.setattr(planner_mod, "check_equivalence", _eq_then_eq)

    async def _summary_stub(*args, **kwargs):
        return "fake summary"

    monkeypatch.setattr(planner_mod, "summarize_relation", _summary_stub)

    table = _make_table_relation(["id", "name", "age"], "T")
    target = _make_relation(["id"], "T")

    config = PlannerConfig(max_iterations=3, max_retries_per_iteration=3)
    await plan([table], target, config)

    # We expect three select_operator calls: iter1-retry0 (dup),
    # iter1-retry1 (success but not equivalent), iter2-retry0 (success
    # AND equivalent). The third call must see an *empty* rejection
    # list — proof that iteration 2 reset.
    assert call_count["n"] == 3
    assert captured_rejections[0] == []
    assert len(captured_rejections[1]) == 1  # duplicate from iter1-retry0
    assert captured_rejections[2] == []  # iteration 2 starts fresh
