"""The real Model: a thin async client over the Anthropic Messages API.

Conversion between the core's vendor-neutral message types and the Anthropic SDK
types is done by module-level pure functions, so the mapping is unit-testable
without a network call or API key. ``AnthropicModel`` itself only wires those
converters to ``AsyncAnthropic.messages.create``.

The model id is a required constructor argument with no default: model ids are
environment-specific and change over time, so the caller must supply a current
one (confirm the active id in the console/dashboard — do not assume a value).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from anthropic import AsyncAnthropic
from anthropic.types import (
    Message as SdkMessage,
)
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types import (
    TextBlock as SdkTextBlock,
)
from anthropic.types import (
    ToolUseBlock as SdkToolUseBlock,
)

from agentique.core.messages import (
    ContentBlock,
    Message,
    ModelResponse,
    OpaqueBlock,
    StopKind,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from agentique.core.tool import ToolSpec

_ContentBlockParam = TextBlockParam | ToolUseBlockParam | ToolResultBlockParam

# How Anthropic's vendor stop reasons map onto the neutral StopKind. Anything not
# listed (a reason the SDK adds later that we have not modeled) becomes ``other``
# and is surfaced by the Runtime rather than folded into a quiet completion.
_STOP_KIND_BY_VENDOR: dict[str, StopKind] = {
    "end_turn": "done",
    "stop_sequence": "done",
    "max_tokens": "length",
    "tool_use": "tool_use",
    "pause_turn": "paused",
    "refusal": "refusal",
}


def _to_content_param(block: ContentBlock) -> _ContentBlockParam:
    """Convert one core content block to its Anthropic request shape.

    An :class:`OpaqueBlock` is replayed verbatim from its ``provider_data`` (which
    is exactly the vendor JSON it was captured from), so a thinking/citation block
    the core does not model round-trips back out unchanged.
    """
    match block:
        case TextBlock(text=text):
            return TextBlockParam(type="text", text=text)
        case ToolUseBlock(id=id, name=name, input=input):
            return ToolUseBlockParam(
                type="tool_use", id=id, name=name, input=dict(input)
            )
        case ToolResultBlock(
            tool_use_id=tool_use_id, content=content, is_error=is_error
        ):
            return ToolResultBlockParam(
                type="tool_result",
                tool_use_id=tool_use_id,
                content=content,
                is_error=is_error,
            )
        case OpaqueBlock(provider_data=provider_data):
            return cast(_ContentBlockParam, dict(provider_data))


def _to_message_param(message: Message) -> MessageParam:
    """Convert one core message to an Anthropic request message."""
    return MessageParam(
        role=message.role,
        content=[_to_content_param(block) for block in message.content],
    )


def _to_tool_param(spec: ToolSpec) -> ToolParam:
    """Convert a core tool declaration to an Anthropic tool definition."""
    return ToolParam(
        name=spec.name,
        description=spec.description,
        input_schema=dict(spec.input_schema),  # type: ignore[typeddict-item]
    )


def _stop_reason(raw: str | None) -> StopReason:
    """Map a vendor stop-reason string to the neutral :class:`StopReason`.

    A ``None`` (which the API uses transiently) becomes a normal ``done``; an
    unrecognised reason is kept as ``other`` with its ``raw`` value preserved.
    """
    if raw is None:
        return StopReason(kind="done", raw="end_turn")
    return StopReason(kind=_STOP_KIND_BY_VENDOR.get(raw, "other"), raw=raw)


def _usage(sdk_usage: object) -> Usage:
    """Read token counts off the SDK usage object, defaulting missing fields to 0.

    ``getattr`` keeps this robust to ``model_construct`` fixtures (which omit
    usage) and to SDK versions that do not report cache tokens.
    """

    def _count(name: str) -> int:
        value = getattr(sdk_usage, name, None)
        return value if isinstance(value, int) else 0

    if sdk_usage is None:
        return Usage()
    return Usage(
        input_tokens=_count("input_tokens"),
        output_tokens=_count("output_tokens"),
        cache_creation_tokens=_count("cache_creation_input_tokens"),
        cache_read_tokens=_count("cache_read_input_tokens"),
    )


def _to_opaque(block: object) -> OpaqueBlock | None:
    """Carry a vendor block the core does not model through as an OpaqueBlock.

    Real SDK blocks (thinking, citations, server-tool results) are Pydantic models
    with a ``type`` and a JSON ``model_dump``; those become an OpaqueBlock so the
    payload survives pause/resume and round-trips back out. A value that is not a
    recognisable block (no ``type``/``model_dump``) is dropped defensively.
    """
    kind = getattr(block, "type", None)
    dump = getattr(block, "model_dump", None)
    if not isinstance(kind, str) or not callable(dump):
        return None
    return OpaqueBlock(kind=kind, provider_data=dump(mode="json"))


def _from_sdk_response(message: SdkMessage) -> ModelResponse:
    """Convert an Anthropic response back into the core's neutral types.

    Text and tool-use blocks map to their modeled IR blocks; every other block
    kind the core does not model (thinking, server-tool results, citations) is
    carried through as an :class:`OpaqueBlock` rather than silently dropped. The
    vendor ``stop_reason`` and ``usage`` are mapped to the neutral
    :class:`StopReason` and :class:`Usage`.
    """
    blocks: list[ContentBlock] = []
    for block in message.content:
        if isinstance(block, SdkTextBlock):
            blocks.append(TextBlock(text=block.text))
        elif isinstance(block, SdkToolUseBlock):
            raw = block.input
            blocks.append(
                ToolUseBlock(
                    id=block.id,
                    name=block.name,
                    input=raw if isinstance(raw, dict) else {},
                )
            )
        else:
            opaque = _to_opaque(block)
            if opaque is not None:
                blocks.append(opaque)
    return ModelResponse(
        message=Message(role="assistant", content=tuple(blocks)),
        stop_reason=_stop_reason(message.stop_reason),
        usage=_usage(getattr(message, "usage", None)),
    )


class AnthropicModel:
    """An async Model backed by the Anthropic Messages API."""

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 4096,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = client if client is not None else AsyncAnthropic()

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[_to_message_param(m) for m in messages],
            tools=[_to_tool_param(t) for t in tools],
        )
        return _from_sdk_response(response)
