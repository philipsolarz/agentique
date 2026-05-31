"""Typed I/O: the engine validates tool arguments before a tool runs and an
agent's final output before it completes, turning a mismatch into a
self-correctable error rather than running/completing with bad data. With no type
declared, nothing changes."""

from collections.abc import Mapping

from pydantic import BaseModel

from agentique.core import (
    Agent,
    Completed,
    Engine,
    Tool,
    ToolResult,
    ToolSpec,
)
from agentique.core.run_context import RunContext
from agentique.testing import StubModel


class _Args(BaseModel):
    value: int


class _CountingTypedTool:
    """A tool whose args are validated against ``_Args`` before it is invoked."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="typed",
            description="needs an int value",
            input_schema={"type": "object"},
            args_model=_Args,
        )

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(content=f"got {arguments['value']}")


def _agent(
    model: StubModel,
    *tools: Tool,
    output_type: type[BaseModel] | None = None,
) -> Agent:
    return Agent(
        name="t",
        instructions="sys",
        model=model,
        tools=tuple(tools),
        output_type=output_type,
    )


async def test_invalid_tool_args_error_without_running_the_tool() -> None:
    tool = _CountingTypedTool()
    model = StubModel(
        [
            StubModel.tool_call("c1", "typed", {}),  # missing required 'value'
            StubModel.text("recovered"),
        ]
    )
    result = await Engine().run(_agent(model, tool), "go")
    assert isinstance(result, Completed)
    assert result.output == "recovered"
    assert tool.calls == []  # the tool never ran with invalid arguments


async def test_valid_tool_args_run_the_tool() -> None:
    tool = _CountingTypedTool()
    model = StubModel(
        [
            StubModel.tool_call("c1", "typed", {"value": 5}),
            StubModel.text("done"),
        ]
    )
    result = await Engine().run(_agent(model, tool), "go")
    assert isinstance(result, Completed)
    assert tool.calls == [{"value": 5}]


class _Out(BaseModel):
    answer: str


async def test_output_type_drives_self_correction_then_completes() -> None:
    model = StubModel(
        [
            StubModel.text("not json at all"),  # fails output_type validation
            StubModel.text('{"answer": "42"}'),  # valid on the retry
        ]
    )
    result = await Engine().run(_agent(model, output_type=_Out), "go")
    assert isinstance(result, Completed)
    assert result.output == '{"answer": "42"}'
    # both turns were used: the first did not complete the run.
    assert result.context.turn == 2


async def test_valid_output_completes_on_first_try() -> None:
    model = StubModel([StubModel.text('{"answer": "ok"}')])
    result = await Engine().run(_agent(model, output_type=_Out), "go")
    assert isinstance(result, Completed)
    assert result.context.turn == 1
