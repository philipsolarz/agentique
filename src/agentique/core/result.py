"""Result: the explicit outcome of a run.

Modeled as a closed union of frozen dataclasses — one variant per terminal
state — so illegal states are unrepresentable: a ``Blocked`` cannot carry a
completion payload, because that field does not exist on it, and callers are
pushed to ``match`` every variant. ``Completed`` and ``Blocked`` carry the
terminal Context for inspection or forking; ``NeedsHuman`` carries a resumable
:class:`Paused` snapshot instead — the value :meth:`Engine.resume` consumes.
"""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from agentique.core.context import Context


@dataclass(frozen=True, slots=True)
class Completed:
    """The agent finished its task. ``output`` is the final assistant text."""

    output: str
    context: Context


@dataclass(frozen=True, slots=True)
class Paused:
    """A self-contained, resumable snapshot of one paused run.

    Carries the immutable ``context`` accumulated up to the pause plus the
    ``pending_tool_use_id`` of the unanswered ``ask_human`` call, so
    :meth:`Engine.resume` can pair the human's answer to the exact call that
    requested it. It is a value of values (no model or tool objects), so a caller
    holding several paused runs can resume any one independently — pause is a
    property of *this* run, not of the conversation as a whole.
    """

    context: Context
    pending_tool_use_id: str


@dataclass(frozen=True, slots=True)
class NeedsHuman:
    """The run paused awaiting human input. ``question`` is what is being asked;
    ``paused`` is the resumable snapshot to hand back to :meth:`Engine.resume`."""

    question: str
    paused: Paused


@dataclass(frozen=True, slots=True)
class Blocked:
    """The run cannot proceed — e.g. a denied permission or an unrecoverable
    tool failure. ``reason`` explains why."""

    reason: str
    context: Context


type Result = Completed | NeedsHuman | Blocked
"""The closed set of run outcomes. Branch on it with an exhaustive ``match``."""
