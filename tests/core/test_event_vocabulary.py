"""The core event vocabulary and the sink seam: events are values, the NullSink
discards, and a collecting sink preserves emission order."""

from agentique.core import (
    Compaction,
    Dispatch,
    ModelCallFinished,
    ModelCallStarted,
    NullSink,
    StopReason,
    ToolCalled,
    TurnBoundary,
    Usage,
)
from agentique.testing import CollectingSink


def test_events_are_frozen_values() -> None:
    started = ModelCallStarted(message_count=2, tool_count=1)
    assert started.message_count == 2
    # equality by value (frozen pydantic dataclass)
    assert ModelCallStarted(message_count=2, tool_count=1) == started


def test_null_sink_discards() -> None:
    sink = NullSink()
    # No state, never raises, returns None.
    assert sink.emit(TurnBoundary(turn=1)) is None


def test_collecting_sink_preserves_order() -> None:
    sink = CollectingSink()
    sink.emit(ModelCallStarted(message_count=1, tool_count=0))
    sink.emit(
        ModelCallFinished(
            stop_reason=StopReason(kind="tool_use", raw="tool_use"),
            usage=Usage(input_tokens=10, output_tokens=4),
            block_count=1,
        )
    )
    sink.emit(ToolCalled(tool_name="echo", is_error=False))
    sink.emit(TurnBoundary(turn=1))
    sink.emit(Compaction(evicted_blocks=2))
    sink.emit(Dispatch(run_id="r2", parent_id="r1", agent_id="planner"))

    kinds = [type(e).__name__ for e in sink.events]
    assert kinds == [
        "ModelCallStarted",
        "ModelCallFinished",
        "ToolCalled",
        "TurnBoundary",
        "Compaction",
        "Dispatch",
    ]
    finished = sink.events[1]
    assert isinstance(finished, ModelCallFinished)
    assert finished.usage.input_tokens == 10
