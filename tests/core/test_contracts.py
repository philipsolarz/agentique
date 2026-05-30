"""Contract-level checks: the public seams compose, and Result is exhaustive.

These exercise the *types*, not behavior — there are no implementations yet.
``_FakeModel`` deliberately uses no inheritance, proving the Model seam is
structural: shape alone makes it a Model. ``_describe`` proves the Result union
is closed by relying on exhaustiveness (``assert_never`` for the impossible arm).
"""

from collections.abc import Sequence
from typing import assert_never

from agentique.core import (
    Agent,
    Blocked,
    Completed,
    Context,
    Message,
    ModelResponse,
    NeedsHuman,
    Result,
    TextBlock,
)
from agentique.core.tool import ToolSpec


class _FakeModel:
    """A structural Model: the right shape, no base class."""

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        return ModelResponse(
            message=Message(role="assistant", content=(TextBlock("ok"),)),
            stop_reason="end_turn",
        )


def test_agent_accepts_structural_model() -> None:
    agent = Agent(name="t", instructions="i", model=_FakeModel())
    assert agent.tools == ()
    assert agent.skills == ()
    assert agent.permissions.allowed_tools is None


def _describe(result: Result) -> str:
    match result:
        case Completed(output=out):
            return f"done:{out}"
        case NeedsHuman(question=q):
            return f"ask:{q}"
        case Blocked(reason=r):
            return f"blocked:{r}"
        case _:
            assert_never(result)


def test_result_union_is_exhaustive() -> None:
    ctx = Context()
    assert _describe(Completed("x", ctx)) == "done:x"
    assert _describe(NeedsHuman("y", ctx)) == "ask:y"
    assert _describe(Blocked("z", ctx)) == "blocked:z"
