"""Capture events: immutability of the snapshot and totality of serialization.

The end-to-end "mutate an argument *after the wrapped call* and the recorded
event is unchanged" property is proven in the wrapper tests (that is where the
``dict(arguments)`` snapshot is taken). Here we cover the events layer itself:
frozen fields, an independently-retained argument copy, and a serializer that
never crashes on a non-JSON-able value.
"""

import json
from dataclasses import FrozenInstanceError

import pytest

from observability.events import (
    ContentBlockSummary,
    ModelCallEvent,
    ToolCallEvent,
    to_jsonl_line,
)


def _tool_event(arguments: dict[str, object]) -> ToolCallEvent:
    return ToolCallEvent(
        tool_name="read_file",
        arguments=arguments,
        result_len=12,
        is_error=False,
        latency_s=0.01,
    )


def test_events_are_frozen() -> None:
    event = _tool_event({"path": "a.txt"})
    with pytest.raises(FrozenInstanceError):
        event.tool_name = "other"  # ty: ignore[invalid-assignment]


def test_argument_copy_is_retained_independently() -> None:
    # The events layer's half of the immutability guarantee: when given a copy,
    # the event keeps it even as the caller's original mapping changes. The
    # wrapper is what makes that copy (proven in the wrapper tests).
    original: dict[str, object] = {"path": "a.txt"}
    event = _tool_event(dict(original))
    original["path"] = "mutated"
    original["added"] = True
    assert event.arguments == {"path": "a.txt"}


def test_to_jsonl_line_is_total_on_non_jsonable_argument() -> None:
    sentinel = object()
    event = _tool_event({"weird": sentinel})

    line = to_jsonl_line(event)

    parsed = json.loads(line)  # valid JSON, did not raise
    assert parsed["kind"] == "tool_call"
    assert parsed["arguments"]["weird"] == repr(sentinel)


def test_model_event_serializes_block_summaries() -> None:
    event = ModelCallEvent(
        system_len=42,
        message_count=3,
        tool_names=("read_file",),
        stop_reason="tool_use",
        blocks=(ContentBlockSummary(kind="tool_use", size=18),),
        latency_s=0.5,
    )

    parsed = json.loads(to_jsonl_line(event))

    assert parsed["kind"] == "model_call"
    assert parsed["stop_reason"] == "tool_use"
    assert parsed["tool_names"] == ["read_file"]
    assert parsed["blocks"] == [{"kind": "tool_use", "size": 18}]
    assert parsed["raised"] is None


def test_failed_model_event_records_exception_type() -> None:
    event = ModelCallEvent(
        system_len=10,
        message_count=1,
        tool_names=(),
        stop_reason=None,
        blocks=(),
        latency_s=0.2,
        raised="ConnectionError",
    )

    parsed = json.loads(to_jsonl_line(event))

    assert parsed["stop_reason"] is None
    assert parsed["raised"] == "ConnectionError"
