"""Result: the explicit outcome of a run.

Modeled as a closed union of frozen dataclasses — one variant per terminal
state — so illegal states are unrepresentable: a ``Blocked`` cannot carry a
completion payload, because that field does not exist on it, and callers are
pushed to ``match`` every variant. Each variant carries the terminal Context so
the caller can inspect, fork, or (for ``NeedsHuman``) later resume the run.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentique.core.context import Context


@dataclass(frozen=True, slots=True)
class Completed:
    """The agent finished its task. ``output`` is the final assistant text."""

    output: str
    context: Context


@dataclass(frozen=True, slots=True)
class NeedsHuman:
    """The run paused awaiting human input. ``question`` is what is being asked;
    ``context`` is the state from which the run can later be resumed."""

    question: str
    context: Context


@dataclass(frozen=True, slots=True)
class Blocked:
    """The run cannot proceed — e.g. a denied permission or an unrecoverable
    tool failure. ``reason`` explains why."""

    reason: str
    context: Context


type Result = Completed | NeedsHuman | Blocked
"""The closed set of run outcomes. Branch on it with an exhaustive ``match``."""
