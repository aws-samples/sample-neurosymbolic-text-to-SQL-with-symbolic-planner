"""Evidence forwarding helper for the BIRD benchmark framework.

BIRD records carry an optional ``evidence`` field that contains hints,
column descriptions, or value mappings intended to help the planner. The
Benchmark_Framework forwards that evidence to ``text_to_sql_planner.main.run``
in one of three shapes (Req 11.1, 11.2, 11.3):

1. **No evidence** — when the evidence string is empty after stripping
   leading/trailing whitespace, the planner is invoked with the bare
   question and ``RunResult.planner_input`` records the question as-is.

2. **Dedicated keyword parameter** — when the planner's signature exposes
   one of the recognised parameter names (``evidence``, ``hint``,
   ``bird_evidence``), the evidence is passed through that keyword and the
   question is passed verbatim through ``question``.
   ``RunResult.planner_input`` records the bare question.

3. **Newline concatenation fallback** — when the planner has no dedicated
   evidence parameter, the evidence is concatenated to the question with
   a single ``\\n`` separator and the combined string is passed through
   ``question``. ``RunResult.planner_input`` records the combined string
   so report consumers can reproduce the planner input byte-for-byte
   (Req 11.4).

The helper deliberately stops short of awaiting the planner. The runner is
responsible for the ``asyncio.wait_for`` timeout wrapper, so this module
only constructs the call and returns the awaitable plus the recorded input
string.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable

# Recognised dedicated parameter names, in order of preference. The first
# name in this tuple that appears in the planner's signature wins, which
# means a planner exposing both ``evidence`` and ``hint`` would route
# through ``evidence``. The order matches the prose in Req 11.3 ("a
# dedicated keyword parameter for evidence or hint text") and gives the
# most BIRD-native name precedence.
_EVIDENCE_PARAM_NAMES: tuple[str, ...] = ("evidence", "hint", "bird_evidence")


def _find_evidence_param(run_callable: Callable[..., Any]) -> str | None:
    """Return the first recognised evidence parameter name on ``run_callable``.

    Returns ``None`` when none of the recognised names appear in the
    callable's signature, or when ``inspect.signature`` cannot inspect the
    callable (for example, a builtin without a signature). A callable that
    accepts arbitrary keyword arguments via ``**kwargs`` does *not* count as
    exposing a dedicated parameter — the design treats only explicitly
    named parameters as a signal that the planner knows what to do with the
    evidence (Req 11.3).
    """
    try:
        signature = inspect.signature(run_callable)
    except (TypeError, ValueError):
        # Builtins or C-implemented callables may refuse signature
        # inspection. Treat them as having no dedicated parameter and fall
        # back to concatenation.
        return None

    parameters = signature.parameters
    for name in _EVIDENCE_PARAM_NAMES:
        param = parameters.get(name)
        if param is None:
            continue
        # Reject ``*args``/``**kwargs`` matches: the design wants an
        # explicitly named parameter.
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        return name
    return None


def forward_call(
    question: str,
    evidence: str,
    run_callable: Callable[..., Any],
    *,
    schema: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[Awaitable[Any], str]:
    """Invoke ``run_callable`` with the question and evidence appropriately.

    Args:
        question: The Test_Case's natural-language question, passed through
            unchanged when a dedicated evidence parameter is found or when
            the evidence is empty.
        evidence: The Test_Case's evidence string, possibly empty or
            whitespace-only. The string is treated as "no evidence" when
            ``evidence.strip()`` is empty (Req 11.2).
        run_callable: The planner callable, typically
            ``text_to_sql_planner.main.run``. Must be an ``async`` callable
            so the returned object is awaitable; ``forward_call`` itself
            does not await it.
        schema: The CREATE-TABLE schema text, forwarded to the planner via
            its required ``schema`` keyword argument.
        extra_kwargs: Optional additional keyword arguments to forward to
            the planner (for example, a ``config`` instance). Passing
            extras here keeps this module decoupled from the planner's
            full signature.

    Returns:
        ``(awaitable, planner_input)`` — the awaitable result of the
        planner call and the exact string that ended up as the
        ``question`` argument, suitable for ``RunResult.planner_input``
        (Req 11.4).
    """
    extras: dict[str, Any] = dict(extra_kwargs) if extra_kwargs else {}

    # Req 11.2: empty / whitespace-only evidence means "invoke the planner
    # with the question alone." We do not even inspect the signature in
    # this branch — there is no evidence to route either way.
    if not evidence.strip():
        awaitable = run_callable(question=question, schema=schema, **extras)
        return awaitable, question

    # Req 11.3 first half: prefer a dedicated keyword parameter when the
    # planner exposes one. The question travels untouched and the evidence
    # rides on its own keyword.
    evidence_param = _find_evidence_param(run_callable)
    if evidence_param is not None:
        # Pre-flight collision check: the recognised parameter name must
        # not also appear in ``extras``. Allowing a silent override would
        # let the runner bypass evidence forwarding by accident.
        if evidence_param in extras:
            raise TypeError(
                f"forward_call: extra_kwargs collides with the planner's "
                f"dedicated evidence parameter {evidence_param!r}"
            )
        kwargs: dict[str, Any] = {
            "question": question,
            "schema": schema,
            evidence_param: evidence,
            **extras,
        }
        awaitable = run_callable(**kwargs)
        return awaitable, question

    # Req 11.3 second half: no dedicated parameter, so concatenate with a
    # single newline and pass the combined string as the question.
    combined = f"{question}\n{evidence}"
    awaitable = run_callable(question=combined, schema=schema, **extras)
    return awaitable, combined


__all__ = ["forward_call"]
