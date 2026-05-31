"""RunContext: the per-call handle the Engine passes to a tool.

A tool receives, alongside its arguments, a ``RunContext`` carrying the run's
*capabilities*: the run id, an event-emit handle, and — when the run is driven by a
Scheduler — a dispatch handle for spawning a child run bound to this one (so the
child records this run as its parent). It is deliberately a *live handle*, not a
serializable value: it never enters the immutable ``Context`` (run *state*) —
capabilities and state are different kinds, kept as different types.

Under a bare :class:`~agentique.core.runtime.Engine` (no Scheduler) ``dispatch`` is
``None``; a tool that needs to dispatch reports a clear error rather than spawning.
``emit`` defaults to the :class:`~agentique.core.events.NullSink`, so a tool can
always emit without checking for a sink.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentique.core.events import EventSink, NullSink

if TYPE_CHECKING:
    from agentique.core.result import Result

type Dispatcher = Callable[[str, str], Awaitable[Result]]
"""Routes a child dispatch through the Scheduler, bound to the current run: given a
target ``agent_id`` and ``prompt``, returns the child's :class:`Result`."""


@dataclass(frozen=True, slots=True)
class RunContext:
    """A tool's handle on its run: identity, event emission, and dispatch."""

    run_id: str = ""
    emit: EventSink = field(default_factory=NullSink)
    dispatch: Dispatcher | None = None

    @classmethod
    def for_test(
        cls,
        *,
        run_id: str = "test",
        emit: EventSink | None = None,
        dispatch: Dispatcher | None = None,
    ) -> RunContext:
        """A bare RunContext for exercising a tool directly in a test."""
        return cls(
            run_id=run_id,
            emit=emit if emit is not None else NullSink(),
            dispatch=dispatch,
        )
