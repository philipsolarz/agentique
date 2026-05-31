"""A deterministic, in-memory :class:`~agentique.core.events.EventSink` for tests.

Collects every emitted event in order so assertions can inspect what a run
produced. The offline counterpart to the dev-only observability recorder, but
working off the core event seam rather than wrapping the Model/Tool seams.
"""

from __future__ import annotations

from agentique.core.events import Event


class CollectingSink:
    """An EventSink that appends every event to ``events`` in emission order."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)
