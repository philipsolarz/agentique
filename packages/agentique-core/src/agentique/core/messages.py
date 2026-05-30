"""Conversation value types exchanged with the model.

These mirror the shape of the Anthropic Messages API (content blocks within
messages) but are defined independently, so the core never depends on a vendor
SDK at the type level. Every type is frozen: a conversation history is a value,
not something mutated in place.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

type Role = Literal["user", "assistant"]
"""Who authored a message. The system prompt is supplied separately to the
model, not modeled as a role here."""

type StopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
]
"""Why the model stopped generating. The Runtime branches on this: ``tool_use``
means tool calls are pending and must be dispatched; the others end the turn.

This mirrors the full set the Anthropic Messages API can return (widened from
the initial four during A2, when the real client surfaced ``pause_turn`` and
``refusal``). Keeping it identical to the vendor set means responses convert
without a lossy remap."""


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A run of natural-language text within a message."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    """The model's request to invoke a tool.

    ``input`` holds the arguments the model produced as arbitrary JSON; the keys
    and value types are not known statically, so values are typed ``object``
    rather than a bare ``Any``.
    """

    id: str
    name: str
    input: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """The outcome of a tool call, fed back to the model on the next turn."""

    tool_use_id: str
    content: str
    is_error: bool = False


type ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock
"""One element of a message body. The union is closed — every block the core
understands appears here — so an exhaustive ``match`` is possible."""


@dataclass(frozen=True, slots=True)
class Message:
    """A single conversational turn: an author and its ordered content blocks."""

    role: Role
    content: tuple[ContentBlock, ...]


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """What a :class:`~agentique.core.model.Model` returns: the assistant
    message together with why generation stopped."""

    message: Message
    stop_reason: StopReason
