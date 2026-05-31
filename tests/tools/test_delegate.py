"""Delegate: one agent dispatching another through the Scheduler, exercised offline.

Scripts both child and parent deterministically with ``StubModel``. The property
under test is *topology through the primitives*: the parent's Delegate tool
dispatches a registered child via ``ctx.dispatch``; the child runs as its own
Scheduler run (recording the parent as its parent), and its real Result flows back
— a completion folds in, a child pause becomes a real parent pause, a block is an
error result.
"""

from agentique.core import (
    Agent,
    Blocked,
    Completed,
    NeedsHuman,
    Permissions,
    RunContext,
    Scheduler,
    Tool,
)
from agentique.core.messages import ModelResponse, ToolResultBlock
from agentique.testing import StubModel
from agentique.tools import AskHuman, Delegate


def _child_agent() -> Agent:
    return Agent(
        name="child",
        instructions="you are the child",
        model=StubModel([StubModel.text("child says hi")]),
    )


def _parent(
    *responses: ModelResponse,
    tools: tuple[Tool, ...],
    permissions: Permissions | None = None,
) -> Agent:
    return Agent(
        name="parent",
        instructions="you are the parent",
        model=StubModel(list(responses)),
        tools=tools,
        permissions=permissions if permissions is not None else Permissions(),
    )


async def test_parent_delegates_and_gets_child_output() -> None:
    delegate = Delegate("child", name="ask_child", description="Delegate to child.")
    parent = _parent(
        StubModel.tool_call("d1", "ask_child", {"prompt": "say hi"}),
        StubModel.text("parent done"),
        tools=(delegate,),
    )
    sched = Scheduler()
    sched.register("child", _child_agent())
    sched.register("parent", parent)

    result = await sched.dispatch("parent", "begin")
    assert isinstance(result, Completed)
    assert result.output == "parent done"

    # the child's output was folded back into the parent conversation.
    tool_results = [
        block
        for message in result.context.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].content == "child says hi"

    # run-lineage: the child run records the parent run as its parent.
    child_run = next(r for r in sched.runs() if r.agent_id == "child")
    parent_run = next(r for r in sched.runs() if r.agent_id == "parent")
    assert child_run.parent_id == parent_run.id


async def test_child_pause_propagates_as_a_real_parent_pause() -> None:
    child = Agent(
        name="child",
        instructions="child",
        model=StubModel(
            [StubModel.tool_call("h1", "ask_human", {"question": "which file?"})]
        ),
        tools=(AskHuman(),),
    )
    delegate = Delegate("child", name="ask_child", description="d")
    parent = _parent(
        StubModel.tool_call("d1", "ask_child", {"prompt": "go"}),
        tools=(delegate,),
    )
    sched = Scheduler()
    sched.register("child", child)
    sched.register("parent", parent)

    result = await sched.dispatch("parent", "begin")
    assert isinstance(result, NeedsHuman)  # not an error string — a real pause
    assert result.question == "which file?"


async def test_delegation_is_permission_gated_like_any_tool() -> None:
    delegate = Delegate("child", name="ask_child", description="delegate")
    parent = _parent(
        StubModel.tool_call("d1", "ask_child", {"prompt": "x"}),
        tools=(delegate,),
        permissions=Permissions.allowlist(()),  # delegation denied
    )
    sched = Scheduler()
    sched.register("child", _child_agent())
    sched.register("parent", parent)

    result = await sched.dispatch("parent", "begin")
    assert isinstance(result, Blocked)
    assert "permission denied" in result.reason


async def test_dispatch_requires_a_scheduler_under_a_bare_engine() -> None:
    delegate = Delegate("child", name="ask_child", description="d")
    result = await delegate(RunContext.for_test(), {"prompt": "hi"})
    assert result.is_error is True
    assert "dispatch requires a scheduler" in result.content


async def test_non_string_prompt_is_reported_as_error() -> None:
    delegate = Delegate("child", name="ask_child", description="d")
    result = await delegate(RunContext.for_test(), {"prompt": 123})
    assert result.is_error is True
    assert "must be a string" in result.content
