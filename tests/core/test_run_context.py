"""RunContext: a tool receives the run handle, and under a bare Engine (no
Scheduler) its dispatch capability is absent."""

from collections.abc import Mapping

from agentique.core import (
    Agent,
    Completed,
    Engine,
    RunContext,
    ToolResult,
    ToolSpec,
    TurnBoundary,
)
from agentique.testing import StubModel


class _ProbeTool:
    """Records the RunContext it was handed."""

    def __init__(self) -> None:
        self.seen: RunContext | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="probe", description="d", input_schema={"type": "object"})

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        self.seen = ctx
        return ToolResult(content="ok")


async def test_tool_receives_run_context_without_dispatch_under_bare_engine() -> None:
    probe = _ProbeTool()
    model = StubModel([StubModel.tool_call("c1", "probe", {}), StubModel.text("done")])
    agent = Agent(name="t", instructions="s", model=model, tools=(probe,))
    result = await Engine().run(agent, "go")
    assert isinstance(result, Completed)
    assert probe.seen is not None
    # a bare Engine has no Scheduler, so there is no dispatch capability
    assert probe.seen.dispatch is None


def test_for_test_factory_defaults() -> None:
    ctx = RunContext.for_test()
    assert ctx.run_id == "test"
    assert ctx.dispatch is None
    # the default sink discards without error
    assert ctx.emit.emit(TurnBoundary(turn=1)) is None
