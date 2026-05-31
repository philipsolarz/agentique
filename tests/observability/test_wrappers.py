"""Recording wrappers: observe-only fidelity, failure capture, snapshot
immutability, and latency that spans only the inner await.

These prove the correctness properties that live at the wrapper layer (the events
layer half was covered in test_events). A ``StubModel``/``EchoTool`` stand in for a
real model/tool so the checks are deterministic and offline.
"""

import asyncio
from collections.abc import Mapping

import pytest

from agentique.core import Agent, Engine
from agentique.core.run_context import RunContext
from agentique.core.tool import ToolResult, ToolSpec
from agentique.testing import EchoTool, StubModel, StubModelExhausted
from observability.events import ModelCallEvent, ToolCallEvent
from observability.recorder import InMemoryRecorder
from observability.wrappers import RecordingModel, RecordingTool


def _script() -> list:
    """One tool-use turn then a terminal text turn — exercises both seams."""
    return [
        StubModel.tool_call("call-1", "echo", {"value": "hi"}),
        StubModel.text("done"),
    ]


async def test_wrapped_run_yields_identical_result() -> None:
    runtime = Engine()

    unwrapped = Agent(
        name="a", instructions="sys", model=StubModel(_script()), tools=(EchoTool(),)
    )
    result_unwrapped = await runtime.run(unwrapped, "go")

    recorder = InMemoryRecorder()
    wrapped = Agent(
        name="a",
        instructions="sys",
        model=RecordingModel(StubModel(_script()), recorder),
        tools=(RecordingTool(EchoTool(), recorder),),
    )
    result_wrapped = await runtime.run(wrapped, "go")

    assert result_wrapped == result_unwrapped
    # And the run was actually observed: two model calls + one tool call.
    kinds = [type(e).__name__ for e in recorder.events]
    assert kinds == ["ModelCallEvent", "ToolCallEvent", "ModelCallEvent"]


async def test_spec_delegates_to_inner() -> None:
    inner = EchoTool()
    wrapped = RecordingTool(inner, InMemoryRecorder())
    assert wrapped.spec == inner.spec


async def test_raised_inner_model_is_recorded_then_reraised() -> None:
    recorder = InMemoryRecorder()
    model = RecordingModel(StubModel([]), recorder)  # exhausted on first call

    with pytest.raises(StubModelExhausted):
        await model.complete(system="s", messages=(), tools=())

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert isinstance(event, ModelCallEvent)
    assert event.raised == "StubModelExhausted"
    assert event.stop_reason is None
    assert event.blocks == ()


async def test_raised_inner_tool_is_recorded_then_reraised() -> None:
    class _BoomTool:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(name="boom", description="", input_schema={})

        async def __call__(
            self, ctx: RunContext, arguments: Mapping[str, object]
        ) -> ToolResult:
            raise RuntimeError("boom")

    recorder = InMemoryRecorder()
    tool = RecordingTool(_BoomTool(), recorder)

    with pytest.raises(RuntimeError):
        await tool(RunContext.for_test(), {"x": 1})

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert isinstance(event, ToolCallEvent)
    assert event.raised == "RuntimeError"
    assert event.result_len is None
    assert event.is_error is None
    assert event.arguments == {"x": 1}


async def test_argument_snapshot_survives_post_call_mutation() -> None:
    recorder = InMemoryRecorder()
    tool = RecordingTool(EchoTool(), recorder)

    arguments: dict[str, object] = {"value": "original"}
    await tool(RunContext.for_test(), arguments)
    arguments["value"] = "mutated"
    arguments["added"] = True

    event = recorder.events[0]
    assert isinstance(event, ToolCallEvent)
    assert event.arguments == {"value": "original"}


async def test_latency_spans_only_the_inner_await() -> None:
    delay = 0.05

    class _SlowTool:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec(name="slow", description="", input_schema={})

        async def __call__(
            self, ctx: RunContext, arguments: Mapping[str, object]
        ) -> ToolResult:
            await asyncio.sleep(delay)
            return ToolResult(content="ok")

    recorder = InMemoryRecorder()
    tool = RecordingTool(_SlowTool(), recorder)

    await tool(RunContext.for_test(), {})

    event = recorder.events[0]
    assert isinstance(event, ToolCallEvent)
    # The recorded span covers the inner await (>= the sleep) and adds little.
    assert event.latency_s >= delay * 0.9
    assert event.latency_s < delay + 0.5
