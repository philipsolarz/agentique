"""Human-in-the-loop: ``ask_human`` pauses to ``NeedsHuman``, ``resume`` continues
that exact run, and resuming replays no prior side effect — exercised offline.
"""

from agentique.core import (
    Agent,
    Blocked,
    Completed,
    Engine,
    Message,
    ModelResponse,
    NeedsHuman,
    StopReason,
)
from agentique.core.messages import ToolResultBlock, ToolUseBlock
from agentique.core.tool import Tool
from agentique.testing import EchoTool, StubModel
from agentique.tools import AskHuman


def _agent(model: StubModel, *tools: Tool) -> Agent:
    return Agent(name="t", instructions="sys", model=model, tools=tuple(tools))


async def test_ask_human_pauses_then_resume_completes() -> None:
    model = StubModel(
        [
            StubModel.tool_call("h1", "ask_human", {"question": "proceed?"}),
            StubModel.text("done"),
        ]
    )
    agent = _agent(model, AskHuman())
    runtime = Engine()

    paused = await runtime.run(agent, prompt="go")
    assert isinstance(paused, NeedsHuman)
    assert paused.question == "proceed?"
    assert paused.paused.pending_tool_use_id == "h1"

    resumed = await runtime.resume(agent, paused.paused, "yes")
    assert isinstance(resumed, Completed)
    assert resumed.output == "done"

    # the human's answer was folded back as the tool_result for the ask_human call.
    tool_results = [
        block
        for message in resumed.context.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_use_id == "h1"
    assert tool_results[0].content == "yes"


async def test_resume_does_not_replay_prior_side_effecting_tools() -> None:
    # A side-effecting tool runs once before the pause; resuming must not re-run it.
    echo = EchoTool()
    model = StubModel(
        [
            StubModel.tool_call("c1", "echo", {"value": "x"}),
            StubModel.tool_call("h1", "ask_human", {"question": "ok?"}),
            StubModel.text("done"),
        ]
    )
    agent = _agent(model, echo, AskHuman())
    runtime = Engine()

    paused = await runtime.run(agent, prompt="go")
    assert isinstance(paused, NeedsHuman)
    assert echo.calls == [{"value": "x"}]  # ran exactly once, before the pause

    resumed = await runtime.resume(agent, paused.paused, "yes")
    assert isinstance(resumed, Completed)
    assert echo.calls == [{"value": "x"}]  # UNCHANGED — no replay across resume


async def test_resume_can_pause_again() -> None:
    model = StubModel(
        [
            StubModel.tool_call("h1", "ask_human", {"question": "first?"}),
            StubModel.tool_call("h2", "ask_human", {"question": "second?"}),
            StubModel.text("done"),
        ]
    )
    agent = _agent(model, AskHuman())
    runtime = Engine()

    first = await runtime.run(agent, prompt="go")
    assert isinstance(first, NeedsHuman)
    assert first.question == "first?"

    second = await runtime.resume(agent, first.paused, "a1")
    assert isinstance(second, NeedsHuman)
    assert second.question == "second?"
    assert second.paused.pending_tool_use_id == "h2"

    done = await runtime.resume(agent, second.paused, "a2")
    assert isinstance(done, Completed)
    assert done.output == "done"


async def test_ask_human_mixed_with_another_call_is_blocked() -> None:
    # ask_human must be the sole tool call in its turn.
    model = StubModel(
        [
            ModelResponse(
                message=Message(
                    role="assistant",
                    content=(
                        ToolUseBlock(id="c1", name="echo", input={"value": "x"}),
                        ToolUseBlock(
                            id="h1", name="ask_human", input={"question": "?"}
                        ),
                    ),
                ),
                stop_reason=StopReason(kind="tool_use", raw="tool_use"),
            ),
        ]
    )
    agent = _agent(model, EchoTool(), AskHuman())
    result = await Engine().run(agent, prompt="go")
    assert isinstance(result, Blocked)
    assert "sole tool call" in result.reason


async def test_two_runs_pause_and_resume_independently() -> None:
    # Per-task pause: two separate runs each park in NeedsHuman; each resumes from
    # its own snapshot, in any order — pause is a property of the run, not global.
    def _m(question: str, out: str) -> StubModel:
        return StubModel(
            [
                StubModel.tool_call("h", "ask_human", {"question": question}),
                StubModel.text(out),
            ]
        )

    runtime = Engine()
    a = _agent(_m("qa", "done-a"), AskHuman())
    b = _agent(_m("qb", "done-b"), AskHuman())

    pa = await runtime.run(a, prompt="a")
    pb = await runtime.run(b, prompt="b")
    assert isinstance(pa, NeedsHuman) and pa.question == "qa"
    assert isinstance(pb, NeedsHuman) and pb.question == "qb"

    # resume b first, then a — order-independent.
    rb = await runtime.resume(b, pb.paused, "yb")
    ra = await runtime.resume(a, pa.paused, "ya")
    assert isinstance(ra, Completed) and ra.output == "done-a"
    assert isinstance(rb, Completed) and rb.output == "done-b"
