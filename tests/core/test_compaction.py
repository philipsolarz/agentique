"""Compaction: under budget is identity, oldest tool results evict first, and the
CompactionMiddleware shrinks a run's context at the pre-model point."""

from collections.abc import Mapping

from agentique.core import (
    Agent,
    Compaction,
    CompactionMiddleware,
    Completed,
    Context,
    Engine,
    EvictOldestToolResults,
    Message,
    TextBlock,
    ToolResult,
    ToolResultBlock,
    ToolSpec,
    estimate_size,
)
from agentique.core.run_context import RunContext
from agentique.testing import CollectingSink, StubModel


def test_under_budget_is_identity() -> None:
    ctx = Context(messages=(Message(role="user", content=(TextBlock("hi"),)),))
    out = EvictOldestToolResults().compact(ctx, budget=1000)
    assert out is ctx  # no copy when nothing to do


def test_evicts_oldest_tool_results_first() -> None:
    comp = EvictOldestToolResults(placeholder="[x]")
    ctx = Context(
        messages=(
            Message(role="user", content=(TextBlock("q"),)),
            Message(role="user", content=(ToolResultBlock("c1", "A" * 10),)),
            Message(role="user", content=(ToolResultBlock("c2", "B" * 10),)),
        ),
        turn=3,
    )
    # size = 1 + 10 + 10 = 21; budget 15 evicts only the oldest (c1).
    out = comp.compact(ctx, budget=15)
    contents = [
        b.content
        for m in out.messages
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]
    assert contents == ["[x]", "B" * 10]
    assert estimate_size(out) <= 15


class _BigTool:
    """Returns a payload far larger than the compaction budget."""

    def __init__(self, size: int) -> None:
        self._payload = "X" * size

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="big", description="d", input_schema={"type": "object"})

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        return ToolResult(content=self._payload)


async def test_compaction_middleware_shrinks_context_and_emits_event() -> None:
    tool = _BigTool(size=200)
    model = StubModel([StubModel.tool_call("c1", "big", {}), StubModel.text("done")])
    agent = Agent(name="t", instructions="s", model=model, tools=(tool,))
    sink = CollectingSink()
    engine = Engine(
        middleware=(
            CompactionMiddleware(EvictOldestToolResults(), budget=50, sink=sink),
        )
    )

    result = await engine.run(agent, "go")
    assert isinstance(result, Completed)
    assert result.output == "done"

    # the big tool result was evicted before the final model call.
    contents = [
        b.content
        for m in result.context.messages
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]
    assert all("X" * 200 not in c for c in contents)
    assert any(isinstance(e, Compaction) for e in sink.events)
