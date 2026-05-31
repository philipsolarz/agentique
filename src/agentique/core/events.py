"""The core event vocabulary and the single sink seam.

A typed, closed set of events describing what happens inside a run — model calls,
tool calls, turn boundaries, compaction, dispatch — plus :class:`EventSink`, the
one mechanism that carries them out of the core. Events are *not* a parallel
observability system: they are exactly what the built-in tracing middleware emits
(see :mod:`agentique.core.middleware`). The core ships the event values and the
emit seam; concrete exporters (e.g. an OpenTelemetry sink) live in a satellite
behind an extra so a base install pulls no telemetry stack.

``EventSink.emit`` is **synchronous** by design: emission happens inline on the
run's own task, so a StubModel-driven run stays fully deterministic — no queue, no
background task, no reordering. A sink that needs to do I/O buffers it itself.
"""

from __future__ import annotations

from typing import Protocol

from pydantic.dataclasses import dataclass

from agentique.core.messages import StopReason, Usage


@dataclass(frozen=True, slots=True)
class ModelCallStarted:
    """A model call is about to be made, with the shape of the request."""

    message_count: int
    tool_count: int


@dataclass(frozen=True, slots=True)
class ModelCallFinished:
    """A model call returned: why it stopped, what it cost, how much it produced."""

    stop_reason: StopReason
    usage: Usage
    block_count: int


@dataclass(frozen=True, slots=True)
class ToolCalled:
    """A tool was dispatched and produced a result (or an error result)."""

    tool_name: str
    is_error: bool


@dataclass(frozen=True, slots=True)
class TurnBoundary:
    """One Runtime turn completed; ``turn`` is the new turn count."""

    turn: int


@dataclass(frozen=True, slots=True)
class Compaction:
    """Context compaction ran before a model call, evicting ``evicted_blocks``."""

    evicted_blocks: int


@dataclass(frozen=True, slots=True)
class Dispatch:
    """One agent dispatched another through the Scheduler, recording lineage."""

    run_id: str
    parent_id: str | None
    agent_id: str


type Event = (
    ModelCallStarted
    | ModelCallFinished
    | ToolCalled
    | TurnBoundary
    | Compaction
    | Dispatch
)
"""The closed set of core events. Branch on it with an exhaustive ``match``."""


class EventSink(Protocol):
    """Where events go. A single synchronous method, called inline on the run."""

    def emit(self, event: Event) -> None: ...


class NullSink:
    """The default sink: discards every event, so the un-instrumented path is free.

    A concrete object (not ``None``) so emit sites stay a plain ``sink.emit(...)``
    with no per-call ``is None`` guard.
    """

    def emit(self, event: Event) -> None:
        return None
