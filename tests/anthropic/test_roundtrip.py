"""Adapter round-trip: the Anthropic adapter is lossless for the blocks the core
models, and carries everything else through the typed OpaqueBlock arm.

The obligation from the IR redesign: ``IR -> vendor -> IR`` must lose nothing for
modeled blocks, and a vendor block the core does not model (thinking, citations,
server-tool results) must survive a round-trip via ``OpaqueBlock.provider_data``
rather than being silently dropped. Usage and the neutral stop reason are mapped,
not discarded.
"""

from anthropic.types import Message as SdkMessage
from anthropic.types import TextBlock as SdkTextBlock
from anthropic.types import ThinkingBlock as SdkThinkingBlock
from anthropic.types import ToolUseBlock as SdkToolUseBlock
from anthropic.types import Usage as SdkUsage

from agentique.anthropic.model import _from_sdk_response, _to_content_param
from agentique.core.messages import OpaqueBlock, TextBlock, ToolUseBlock


def test_modeled_blocks_survive_ir_to_vendor_to_ir() -> None:
    sdk = SdkMessage.model_construct(
        content=[
            SdkTextBlock.model_construct(text="hello"),
            SdkToolUseBlock.model_construct(id="t1", name="f", input={"k": "v"}),
        ],
        stop_reason="tool_use",
    )
    ir = _from_sdk_response(sdk)
    assert ir.message.content == (
        TextBlock("hello"),
        ToolUseBlock(id="t1", name="f", input={"k": "v"}),
    )
    # IR -> vendor: the modeled blocks convert back to their exact request shapes.
    params = [_to_content_param(b) for b in ir.message.content]
    assert params == [
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "t1", "name": "f", "input": {"k": "v"}},
    ]


def test_unmodeled_block_round_trips_as_opaque() -> None:
    thinking = SdkThinkingBlock.model_construct(
        type="thinking", thinking="let me reason", signature="sig-123"
    )
    sdk = SdkMessage.model_construct(
        content=[thinking, SdkTextBlock.model_construct(text="answer")],
        stop_reason="end_turn",
    )
    ir = _from_sdk_response(sdk)

    # The thinking block is carried through, not dropped, with its raw payload.
    assert isinstance(ir.message.content[0], OpaqueBlock)
    opaque = ir.message.content[0]
    assert opaque.kind == "thinking"
    assert opaque.provider_data == {
        "type": "thinking",
        "thinking": "let me reason",
        "signature": "sig-123",
    }
    assert ir.message.content[1] == TextBlock("answer")

    # IR -> vendor: the opaque block replays verbatim from its provider_data.
    assert _to_content_param(opaque) == opaque.provider_data


def test_usage_is_extracted_from_the_response() -> None:
    sdk = SdkMessage.model_construct(
        content=[SdkTextBlock.model_construct(text="hi")],
        stop_reason="end_turn",
        usage=SdkUsage.model_construct(
            input_tokens=11,
            output_tokens=7,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=5,
        ),
    )
    usage = _from_sdk_response(sdk).usage
    assert (usage.input_tokens, usage.output_tokens) == (11, 7)
    assert (usage.cache_creation_tokens, usage.cache_read_tokens) == (3, 5)


def test_unknown_vendor_stop_reason_becomes_other_keeping_raw() -> None:
    sdk = SdkMessage.model_construct(content=[], stop_reason="model_context_window")
    reason = _from_sdk_response(sdk).stop_reason
    assert reason.kind == "other"
    assert reason.raw == "model_context_window"
