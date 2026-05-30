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

from agentique.core.messages import (
    ContentBlock,
    Message,
    ModelResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agentique.core.tool import ToolSpec

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

_ContentBlockParam = TextBlockParam | ToolUseBlockParam | ToolResultBlockParam


def _to_content_param(block: ContentBlock) -> _ContentBlockParam:
    """Convert one core content block to its Anthropic request shape."""
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


def _from_sdk_response(message: SdkMessage) -> ModelResponse:
    """Convert an Anthropic response back into the core's types.

    Only text and tool-use blocks are surfaced; other block kinds (thinking,
    server-tool results) are not enabled by this client and are dropped. The
    response ``stop_reason`` shares the core's :data:`StopReason` literal set, so
    it maps across directly; a ``None`` (which the API uses transiently) becomes
    ``end_turn``.
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
    return ModelResponse(
        message=Message(role="assistant", content=tuple(blocks)),
        stop_reason=message.stop_reason or "end_turn",
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
