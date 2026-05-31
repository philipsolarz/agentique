"""The middleware onion: an empty chain is a pass-through, layers nest
outer→inner inbound / inner→outer outbound, and the built-in TracingMiddleware
emits the event vocabulary without changing the run's Result."""

from typing import Any

from agentique.core import (
    Agent,
    Completed,
    Engine,
    ModelCallFinished,
    ModelCallStarted,
    ModelPoint,
    Point,
    ToolCalled,
    TracingMiddleware,
    TurnBoundary,
)
from agentique.core.middleware import CallNext
from agentique.testing import CollectingSink, EchoTool, StubModel


def _agent(model: StubModel, *tools: Any) -> Agent:
    return Agent(name="t", instructions="sys", model=model, tools=tuple(tools))


class _Recorder:
    """Records its label on the way in and out of the model point only, so the
    nesting order is legible for a text-only run."""

    def __init__(self, label: str, log: list[str]) -> None:
        self._label = label
        self._log = log

    async def handle(self, point: Point, call_next: CallNext) -> Any:
        if isinstance(point, ModelPoint):
            self._log.append(f"{self._label}>")
            result = await call_next()
            self._log.append(f"{self._label}<")
            return result
        return await call_next()


async def test_empty_chain_matches_passthrough_result() -> None:
    script = [StubModel.text("answer")]
    bare = await Engine().run(_agent(StubModel(script)), "go")
    explicit_empty = await Engine(middleware=()).run(_agent(StubModel(script)), "go")
    assert isinstance(bare, Completed)
    assert bare == explicit_empty


async def test_layers_nest_outer_to_inner_then_unwind() -> None:
    log: list[str] = []
    engine = Engine(middleware=(_Recorder("A", log), _Recorder("B", log)))
    result = await engine.run(_agent(StubModel([StubModel.text("hi")])), "go")
    assert isinstance(result, Completed)
    # A is outermost: inbound A then B, outbound B then A.
    assert log == ["A>", "B>", "B<", "A<"]


async def test_tracing_emits_events_without_changing_result() -> None:
    script = [
        StubModel.tool_call("c1", "echo", {"value": "hi"}),
        StubModel.text("done"),
    ]
    untraced = await Engine().run(_agent(StubModel(script), EchoTool()), "go")

    sink = CollectingSink()
    traced = await Engine(middleware=(TracingMiddleware(sink),)).run(
        _agent(StubModel(script), EchoTool()), "go"
    )

    assert traced == untraced
    kinds = [type(e).__name__ for e in sink.events]
    # two model calls (tool turn + final), one tool call, and a turn boundary for
    # the continuing turn.
    assert kinds.count("ModelCallStarted") == 2
    assert kinds.count("ModelCallFinished") == 2
    assert "ToolCalled" in kinds
    assert "TurnBoundary" in kinds

    tool_events = [e for e in sink.events if isinstance(e, ToolCalled)]
    assert tool_events == [ToolCalled(tool_name="echo", is_error=False)]
    # the first finished event carries the tool_use stop reason and the usage.
    finished = [e for e in sink.events if isinstance(e, ModelCallFinished)]
    assert finished[0].stop_reason.kind == "tool_use"
    started = [e for e in sink.events if isinstance(e, ModelCallStarted)]
    assert started[0].tool_count == 1
    assert isinstance(TurnBoundary(turn=1), TurnBoundary)
