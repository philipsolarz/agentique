"""Delegate: one agent spawning another as a granted tool, exercised offline.

Uses the shipped ``agentique.testing.StubModel`` (a dev dependency of this
package's tests) to script both the child and the parent deterministically. The
key property under test is *topology through the existing primitives*: a parent
Runtime dispatches a Delegate tool whose child runs on its own Runtime, with the
child's output folded back into the parent conversation.
"""

from agentique.core import Agent, Blocked, Completed, Permissions, Runtime
from agentique.core.messages import ToolResultBlock
from agentique.testing import StubModel
from agentique.tools import Delegate


def _child_agent() -> Agent:
    # the child simply answers in one turn.
    return Agent(
        name="child",
        instructions="you are the child",
        model=StubModel([StubModel.text("child says hi")]),
    )


async def test_parent_delegates_and_gets_child_output() -> None:
    delegate = Delegate(
        _child_agent(),
        name="ask_child",
        description="Delegate a task to the child agent.",
    )
    parent = Agent(
        name="parent",
        instructions="you are the parent",
        model=StubModel(
            [
                StubModel.tool_call("d1", "ask_child", {"prompt": "say hi"}),
                StubModel.text("parent done"),
            ]
        ),
        tools=(delegate,),
    )
    result = await Runtime().run(parent, prompt="begin")
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
    assert tool_results[0].is_error is False


async def test_delegation_is_permission_gated_like_any_tool() -> None:
    delegate = Delegate(_child_agent(), name="ask_child", description="delegate")
    parent = Agent(
        name="parent",
        instructions="parent",
        model=StubModel([StubModel.tool_call("d1", "ask_child", {"prompt": "x"})]),
        tools=(delegate,),
        permissions=Permissions(allowed_tools=frozenset()),  # delegation denied
    )
    result = await Runtime().run(parent, prompt="begin")
    assert isinstance(result, Blocked)
    assert "permission denied" in result.reason


async def test_non_string_prompt_is_reported_as_error() -> None:
    delegate = Delegate(_child_agent(), name="ask_child", description="d")
    result = await delegate({"prompt": 123})
    assert result.is_error is True
    assert "must be a string" in result.content
