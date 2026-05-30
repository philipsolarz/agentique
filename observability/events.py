"""Typed capture events: immutable snapshots of what crossed the Model and Tool
seams on one run, plus the run-level rollup.

Every event is a frozen dataclass holding a snapshot taken *at record time* —
sizes and counts are computed eagerly and argument mappings are copied, so a
later mutation of something the Runtime reused cannot reach back and change what
was recorded. The union of per-call events is named ``CallEvent`` (not a bare
``Event``, which reads as :class:`asyncio.Event`).

Serialization is **total**: :func:`to_jsonl_line` passes ``default=repr`` so an
arbitrary, non-JSON-able tool argument is rendered rather than crashing capture.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContentBlockSummary:
    """The kind and size of one content block returned by the model. ``size`` is a
    character count (text length, or the length of the argument repr for a
    tool-use block) — diagnostic, not the block's full contents."""

    kind: str
    size: int


@dataclass(frozen=True, slots=True)
class ModelCallEvent:
    """A snapshot of one ``Model.complete`` call.

    On success ``stop_reason``/``blocks`` describe the response and ``raised`` is
    ``None``; on failure ``raised`` holds the exception type name and the response
    fields are left empty.
    """

    system_len: int
    message_count: int
    tool_names: tuple[str, ...]
    stop_reason: str | None
    blocks: tuple[ContentBlockSummary, ...]
    latency_s: float
    raised: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """A snapshot of one ``Tool.__call__`` invocation.

    ``arguments`` is a shallow copy taken at record time. On failure ``raised``
    holds the exception type name and ``result_len``/``is_error`` are ``None``.
    """

    tool_name: str
    arguments: Mapping[str, object]
    result_len: int | None
    is_error: bool | None
    latency_s: float
    raised: str | None = None


type CallEvent = ModelCallEvent | ToolCallEvent
"""One recorded seam crossing. Closed union — an exhaustive ``match`` is possible."""


@dataclass(frozen=True, slots=True)
class Turn:
    """One Runtime turn: the model call that opened it and the tool calls it
    triggered (the tool calls folded back as the next user message)."""

    model: ModelCallEvent
    tools: tuple[ToolCallEvent, ...]


def group_into_turns(events: Sequence[CallEvent]) -> list[Turn]:
    """Group a flat event stream into turns. Each model call opens a turn; the tool
    calls until the next model call attach to it. A leading tool call with no model
    call yet is ignored — the Runtime never produces one. Shared so the digest and
    the anomaly detector number turns identically."""
    turns: list[Turn] = []
    model: ModelCallEvent | None = None
    tools: list[ToolCallEvent] = []
    for event in events:
        if isinstance(event, ModelCallEvent):
            if model is not None:
                turns.append(Turn(model, tuple(tools)))
            model = event
            tools = []
        elif isinstance(event, ToolCallEvent) and model is not None:
            tools.append(event)
    if model is not None:
        turns.append(Turn(model, tuple(tools)))
    return turns


@dataclass(frozen=True, slots=True)
class RunRecord:
    """The run-level rollup that serializes to ``manifest.json``. ``outcome`` is the
    terminal :class:`~agentique.core.result.Result` variant name; ``turns`` is the
    model-call count (one per Runtime turn) and ``tool_calls`` the tool-invocation
    count; ``anomalies`` is the list of contract gaps the run surfaced."""

    scenario: str
    outcome: str
    turns: int
    tool_calls: int
    wall_time_s: float
    anomalies: tuple[str, ...]


def run_record_to_dict(record: RunRecord) -> dict[str, object]:
    """Render a :class:`RunRecord` as the ``manifest.json`` shape. Kept beside the
    type so the two stay in sync."""
    return {
        "scenario": record.scenario,
        "outcome": record.outcome,
        "turns": record.turns,
        "tool_calls": record.tool_calls,
        "wall_time_s": record.wall_time_s,
        "anomalies": list(record.anomalies),
    }


def _event_to_dict(event: CallEvent) -> dict[str, object]:
    """Render one event as a JSON-shaped dict with a ``kind`` discriminator."""
    match event:
        case ModelCallEvent():
            return {
                "kind": "model_call",
                "system_len": event.system_len,
                "message_count": event.message_count,
                "tool_names": list(event.tool_names),
                "stop_reason": event.stop_reason,
                "blocks": [{"kind": b.kind, "size": b.size} for b in event.blocks],
                "latency_s": event.latency_s,
                "raised": event.raised,
            }
        case ToolCallEvent():
            return {
                "kind": "tool_call",
                "tool_name": event.tool_name,
                "arguments": dict(event.arguments),
                "result_len": event.result_len,
                "is_error": event.is_error,
                "latency_s": event.latency_s,
                "raised": event.raised,
            }


def to_jsonl_line(event: CallEvent) -> str:
    """Serialize one event to a single JSON line. Total: ``default=repr`` ensures a
    value with no JSON encoder is rendered as its repr rather than raising."""
    return json.dumps(_event_to_dict(event), default=repr)
