"""The Runtime loop, exercised offline with a private inline Model stub."""

from _stub import StubModel
from agentique.core import Agent, Blocked, Completed, Runtime


def _agent(model: StubModel) -> Agent:
    return Agent(name="t", instructions="be helpful", model=model)


async def test_single_turn_completes_with_text() -> None:
    agent = _agent(StubModel([StubModel.text("the answer is 42")]))
    result = await Runtime().run(agent, prompt="what is the answer?")
    assert isinstance(result, Completed)
    assert result.output == "the answer is 42"


async def test_completed_context_records_the_exchange() -> None:
    agent = _agent(StubModel([StubModel.text("hi")]))
    result = await Runtime().run(agent, prompt="hello")
    assert isinstance(result, Completed)
    # one user prompt + one assistant reply, and exactly one model call (turn).
    assert result.context.turn == 1
    assert len(result.context.messages) == 2
    assert result.context.messages[0].role == "user"
    assert result.context.messages[1].role == "assistant"


async def test_zero_max_turns_blocks_before_calling_model() -> None:
    agent = _agent(StubModel([]))  # no scripted responses; must not be called
    result = await Runtime(max_turns=0).run(agent, prompt="hello")
    assert isinstance(result, Blocked)
    assert "max_turns" in result.reason
    assert result.context.turn == 0
