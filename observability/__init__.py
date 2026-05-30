"""Agentique observability: a dev-only capture layer that wraps the ``Model`` and
``Tool`` seams to record what a real run does, without altering its behavior.

First-class, tested library code that is deliberately *not* part of the shipped
``agentique`` distribution yet — the capture format is still moving. It imports
core types but is never imported by core. Promotion to ``agentique.testing`` is an
earned-later move, once the format stabilizes.
"""

from observability.anomalies import detect_anomalies, standing_notes
from observability.digest import render_digest
from observability.events import (
    CallEvent,
    ContentBlockSummary,
    ModelCallEvent,
    RunRecord,
    ToolCallEvent,
    Turn,
    group_into_turns,
    run_record_to_dict,
    to_jsonl_line,
)
from observability.instrument import instrument_agent
from observability.recorder import InMemoryRecorder, Recorder
from observability.wrappers import RecordingModel, RecordingTool
from observability.writer import build_run_record, write_run

__all__ = [
    "CallEvent",
    "ContentBlockSummary",
    "InMemoryRecorder",
    "ModelCallEvent",
    "Recorder",
    "RecordingModel",
    "RecordingTool",
    "RunRecord",
    "ToolCallEvent",
    "Turn",
    "build_run_record",
    "detect_anomalies",
    "group_into_turns",
    "instrument_agent",
    "render_digest",
    "run_record_to_dict",
    "standing_notes",
    "to_jsonl_line",
    "write_run",
]
