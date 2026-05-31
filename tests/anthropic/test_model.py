"""Pure-converter tests for the Anthropic client (no network, no API key).

Exercises the request/response mapping functions directly. SDK response objects
are built with ``model_construct`` so fixtures stay minimal while remaining real
instances (so the converter's ``isinstance`` checks behave as in production).
"""

from anthropic.types import Message as SdkMessage
from anthropic.types import TextBlock as SdkTextBlock
from anthropic.types import ToolUseBlock as SdkToolUseBlock

from agentique.anthropic.model import (
    _from_sdk_response,
    _to_content_param,
    _to_message_param,
    _to_tool_param,
)
from agentique.core.messages import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agentique.core.tool import ToolSpec


def test_to_content_param_text() -> None:
    assert _to_content_param(TextBlock("hi")) == {"type": "text", "text": "hi"}


def test_to_content_param_tool_use() -> None:
    block = ToolUseBlock(id="t1", name="f", input={"a": 1})
    assert _to_content_param(block) == {
        "type": "tool_use",
        "id": "t1",
        "name": "f",
        "input": {"a": 1},
    }


def test_to_content_param_tool_result() -> None:
    block = ToolResultBlock(tool_use_id="t1", content="out", is_error=True)
    assert _to_content_param(block) == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "out",
        "is_error": True,
    }


def test_to_message_param_roundtrips_role_and_blocks() -> None:
    msg = Message(role="user", content=(TextBlock("a"), TextBlock("b")))
    param = _to_message_param(msg)
    assert param["role"] == "user"
    assert param["content"] == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_to_tool_param() -> None:
    spec = ToolSpec(name="t", description="d", input_schema={"type": "object"})
    assert _to_tool_param(spec) == {
        "name": "t",
        "description": "d",
        "input_schema": {"type": "object"},
    }


def test_from_sdk_response_extracts_text_and_tool_use() -> None:
    sdk = SdkMessage.model_construct(
        content=[
            SdkTextBlock.model_construct(text="hello"),
            SdkToolUseBlock.model_construct(id="t1", name="f", input={"k": "v"}),
        ],
        stop_reason="tool_use",
    )
    result = _from_sdk_response(sdk)
    assert result.stop_reason.kind == "tool_use"
    assert result.stop_reason.raw == "tool_use"
    assert result.message.role == "assistant"
    assert result.message.content == (
        TextBlock("hello"),
        ToolUseBlock(id="t1", name="f", input={"k": "v"}),
    )


def test_from_sdk_response_defaults_none_stop_reason() -> None:
    sdk = SdkMessage.model_construct(content=[], stop_reason=None)
    reason = _from_sdk_response(sdk).stop_reason
    assert reason.kind == "done"
    assert reason.raw == "end_turn"


def test_from_sdk_response_drops_unknown_blocks() -> None:
    sdk = SdkMessage.model_construct(
        content=[object(), SdkTextBlock.model_construct(text="kept")],
        stop_reason="end_turn",
    )
    result = _from_sdk_response(sdk)
    assert result.message.content == (TextBlock("kept"),)
