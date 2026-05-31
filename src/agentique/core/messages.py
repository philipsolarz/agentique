"""Conversation value types exchanged with the model — a neutral IR.

These are a genuinely *provider-neutral* intermediate representation: they model
the shapes every chat model shares (text, tool calls, tool results) without
binding to any one vendor. A provider that emits blocks the core does not model
(thinking/reasoning, citations, server-tool results, cache_control) carries them
through the explicit :class:`OpaqueBlock` arm, so nothing is silently dropped and
a paused run can be serialized and resumed losslessly. Bidirectional adapters to
a concrete vendor SDK live in that provider's package (``agentique.anthropic``),
never here.

Every type is a frozen Pydantic dataclass: a conversation history is a value, not
something mutated in place, and being Pydantic it is JSON-serializable (which is
what makes durable pause/resume free). Pydantic dataclasses keep positional
construction (``TextBlock("hi")``), ``dataclasses.replace``, and ``match`` — so
they are a drop-in for the stdlib dataclasses they replace, with validation and
serialization added.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from typing import Literal

from pydantic import JsonValue
from pydantic.dataclasses import dataclass

type Role = Literal["user", "assistant"]
"""Who authored a message. The system prompt is supplied separately to the
model, not modeled as a role here."""

type StopKind = Literal[
    "done",
    "tool_use",
    "length",
    "refusal",
    "paused",
    "other",
]
"""The neutral reason a model stopped generating, abstracted from any vendor's
vocabulary: ``done`` (a normal end), ``tool_use`` (tool calls are pending and
must be dispatched), ``length`` (hit a token cap), ``refusal``, ``paused`` (the
provider paused mid-turn — e.g. a server-side tool), and ``other`` for anything
the core does not recognise. The Engine branches on ``tool_use``; anything it
cannot act on it surfaces explicitly rather than folding into a quiet success."""


@dataclass(frozen=True, slots=True)
class StopReason:
    """Why the model stopped, as the neutral :data:`StopKind` plus the ``raw``
    vendor string it was mapped from — so a provider's exact reason is never lost,
    even when the core treats several vendor reasons as one neutral kind."""

    kind: StopKind
    raw: str


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


@dataclass(frozen=True, slots=True)
class OpaqueBlock:
    """A vendor block the core does not model, carried through verbatim.

    ``kind`` is the provider's block type (e.g. ``"thinking"``) and
    ``provider_data`` is its raw payload. The payload is constrained to
    :data:`~pydantic.JsonValue` so it is always JSON-serializable — the property
    durable pause/resume relies on. The core never inspects it; the originating
    provider adapter round-trips it back out unchanged.
    """

    kind: str
    provider_data: Mapping[str, JsonValue]


type ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | OpaqueBlock
"""One element of a message body. The union is closed — every block the core
understands, plus :class:`OpaqueBlock` as the typed catch-all — so an exhaustive
``match`` is possible and no provider block is ever silently discarded."""


@dataclass(frozen=True, slots=True)
class Message:
    """A single conversational turn: an author and its ordered content blocks."""

    role: Role
    content: tuple[ContentBlock, ...]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one model response. No prices live in the core — cost
    is derived from a pricing table outside it. Cache fields default to zero so a
    provider that does not report caching needs no special-casing."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """What a :class:`~agentique.core.model.Model` returns: the assistant message,
    why generation stopped, and the token usage it consumed."""

    message: Message
    stop_reason: StopReason
    usage: Usage = field(default_factory=Usage)
