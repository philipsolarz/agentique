"""The recording seam.

``Recorder`` is the one-method sink the recording wrappers depend on — a Protocol,
not a concrete class, so the wrappers compose against a seam (as the rest of the
framework does) and a test double is trivial. ``record`` is **synchronous and
in-memory by contract**: it must never do inline I/O, which would block the event
loop and corrupt the latency numbers. File output happens once, after the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from observability.events import CallEvent


class Recorder(Protocol):
    """A synchronous sink for capture events."""

    def record(self, event: CallEvent) -> None:
        """Append ``event`` to the capture. Must not block or do I/O."""
        ...


class InMemoryRecorder:
    """The default ``Recorder``: collects events in a list for post-run output."""

    def __init__(self) -> None:
        self._events: list[CallEvent] = []

    def record(self, event: CallEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> tuple[CallEvent, ...]:
        """An immutable view of everything recorded so far, in order."""
        return tuple(self._events)


if TYPE_CHECKING:
    # Confirm the concrete recorder satisfies the Protocol, checked by ``ty``.
    _: Recorder = InMemoryRecorder()
