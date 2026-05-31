"""Context: ephemeral working state for a single run.

Holds the conversation as it accumulates plus the turn counter. Deliberately
minimal and a *distinct type* from :class:`~agentique.core.memory.Memory`
(durable), so ephemeral state cannot be persisted by accident.

Two deliberate simplifications, flagged for review:

* **Tool results** are represented as ``ToolResultBlock`` entries inside the
  message history (the Anthropic-native shape), not as a separate field — this
  avoids a second source of truth for the same data.
* **The stop condition** is owned by the Runtime (per the agreed Agent/Runtime
  split), not stored here. Context records *what happened*; the Runtime decides
  *when to stop*.
"""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from agentique.core.messages import Message


@dataclass(frozen=True, slots=True)
class Context:
    """An immutable snapshot of a single run's working state.

    The Runtime threads a fresh Context through each step rather than mutating
    one in place, keeping run state a value like everything else in the core.
    """

    messages: tuple[Message, ...] = ()
    turn: int = 0
