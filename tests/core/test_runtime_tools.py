"""Runtime tool dispatch + permission enforcement, exercised offline."""

from collections.abc import Mapping

from agentique.core import Agent, Blocked, Completed, Permissions, Runtime
from agentique.core.messages import ToolResultBlock
from agentique.core.run_context import RunContext
from agentique.core.tool import ToolResult, ToolSpec
from agentique.testing import EchoTool, StubModel


def _agent(model: StubModel, tool: EchoTool, permissions: Permissions) -> Agent:
    return Agent(
        name="t",
        instructions="use tools",
        model=model,
        tools=(tool,),
        permissions=permissions,
    )


async def test_dispatches_tool_then_completes() -> None:
    tool = EchoTool()
    model = StubModel(
        [
            StubModel.tool_call("c1", "echo", {"value": "ping"}),
            StubModel.text("done: ping"),
        ]
    )
    result = await Runtime().run(_agent(model, tool, Permissions()), prompt="go")
    assert isinstance(result, Completed)
    assert result.output == "done: ping"
    # the tool actually ran with the model-supplied arguments...
    assert tool.calls == [{"value": "ping"}]
    # ...and its result was folded back as a tool_result block for the next turn.
    tool_results = [
        block
        for message in result.context.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_use_id == "c1"
    assert tool_results[0].content == "ping"


async def test_denied_permission_blocks_without_running_tool() -> None:
    tool = EchoTool()
    model = StubModel([StubModel.tool_call("c1", "echo", {"value": "x"})])
    permissions = Permissions.allowlist(())  # nothing allowed
    result = await Runtime().run(_agent(model, tool, permissions), prompt="go")
    assert isinstance(result, Blocked)
    assert "permission denied" in result.reason
    assert tool.calls == []


async def test_allowlisted_tool_is_dispatched() -> None:
    tool = EchoTool()
    model = StubModel(
        [
            StubModel.tool_call("c1", "echo", {"value": "ok"}),
            StubModel.text("fin"),
        ]
    )
    permissions = Permissions.allowlist({"echo"})
    result = await Runtime().run(_agent(model, tool, permissions), prompt="go")
    assert isinstance(result, Completed)
    assert tool.calls == [{"value": "ok"}]


async def test_unknown_tool_blocks() -> None:
    tool = EchoTool(name="echo")
    model = StubModel([StubModel.tool_call("c1", "nonexistent", {})])
    result = await Runtime().run(_agent(model, tool, Permissions()), prompt="go")
    assert isinstance(result, Blocked)
    assert "unknown tool" in result.reason


async def test_tool_error_is_reported_to_model_not_fatal() -> None:
    tool = EchoTool(is_error=True)
    model = StubModel(
        [
            StubModel.tool_call("c1", "echo", {"value": "boom"}),
            StubModel.text("recovered"),
        ]
    )
    result = await Runtime().run(_agent(model, tool, Permissions()), prompt="go")
    # a tool returning an error result is fed back, and the run continues.
    assert isinstance(result, Completed)
    assert result.output == "recovered"
    tool_results = [
        block
        for message in result.context.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert tool_results[0].is_error is True


class _RaisingTool:
    """A Tool whose ``__call__`` raises — to prove a raise is recoverable."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="boom",
            description="always raises",
            input_schema={"type": "object", "properties": {}},
        )

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        raise RuntimeError("kaboom")


async def test_tool_that_raises_is_recoverable_not_fatal() -> None:
    # A tool that *raises* (rather than returning is_error) must be folded into an
    # error result the model can recover from, not propagated out of run().
    model = StubModel(
        [
            StubModel.tool_call("c1", "boom", {}),
            StubModel.text("recovered"),
        ]
    )
    agent = Agent(name="t", instructions="x", model=model, tools=(_RaisingTool(),))
    result = await Runtime().run(agent, prompt="go")
    assert isinstance(result, Completed)
    assert result.output == "recovered"
    tool_results = [
        block
        for message in result.context.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert tool_results[0].is_error is True
    assert "raised" in tool_results[0].content
