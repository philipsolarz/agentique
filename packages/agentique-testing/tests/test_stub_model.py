"""StubModel: deterministic replay and call recording, fully offline."""

import pytest
from agentique.core import Message, ModelResponse, TextBlock
from agentique.core.tool import ToolSpec
from agentique.testing import StubModel, StubModelExhausted


def _response(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message(role="assistant", content=(TextBlock(text),)),
        stop_reason="end_turn",
    )


async def test_replays_responses_in_order() -> None:
    model = StubModel([_response("one"), _response("two")])
    first = await model.complete(system="s", messages=[], tools=[])
    second = await model.complete(system="s", messages=[], tools=[])
    assert isinstance(first.message.content[0], TextBlock)
    assert first.message.content[0].text == "one"
    assert isinstance(second.message.content[0], TextBlock)
    assert second.message.content[0].text == "two"


async def test_records_each_call() -> None:
    model = StubModel([_response("x")])
    spec = ToolSpec(name="t", description="d", input_schema={})
    msgs = [Message(role="user", content=(TextBlock("hi"),))]
    await model.complete(system="sys", messages=msgs, tools=[spec])
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call.system == "sys"
    assert call.messages == tuple(msgs)
    assert call.tools == (spec,)


async def test_raises_when_exhausted() -> None:
    model = StubModel([_response("only")])
    await model.complete(system="s", messages=[], tools=[])
    with pytest.raises(StubModelExhausted):
        await model.complete(system="s", messages=[], tools=[])
