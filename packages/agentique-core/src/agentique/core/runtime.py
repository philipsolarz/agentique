"""The Runtime: the engine that drives a declarative Agent to a Result.

The Agent declares *what* it is (role, model, tools, skills, permissions); the
Runtime owns *how a run proceeds* — assemble the conversation, call the model,
dispatch any tool calls, decide when the run is done, emit a Result. Run-control
(the turn limit and the stop decision) lives here, not on the Agent, per the
agreed Agent/Runtime split.

The loop (A3 + A4):

1. Guard the turn limit; ``Blocked`` if exceeded.
2. Call the model with the conversation so far and the agent's tool specs.
3. If the model requested tools (``stop_reason == "tool_use"``): enforce the
   agent's permissions, dispatch each allowed tool, fold the results back as a
   user message, and loop.
4. Otherwise the turn is terminal: ``Completed`` with the assistant's text.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from agentique.core.agent import Agent, Permissions
from agentique.core.context import Context
from agentique.core.messages import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agentique.core.result import Blocked, Completed, Result
from agentique.core.tool import Tool


def _final_text(message: Message) -> str:
    """Join the text blocks of an assistant message into the run's output.

    Derived inline rather than by importing the ``ExtractText`` skill: skills live
    in a satellite package that depends on core, so core cannot depend back on
    them. A deliberate two-line duplication, not a missed reuse.
    """
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))


def _is_permitted(permissions: Permissions, tool_name: str) -> bool:
    """Whether ``tool_name`` may be dispatched under ``permissions``.

    ``allowed_tools is None`` means "every tool the agent holds is permitted";
    otherwise the name must appear in the allowlist.
    """
    if permissions.allowed_tools is None:
        return True
    return tool_name in permissions.allowed_tools


@dataclass(frozen=True, slots=True)
class Runtime:
    """Drives one Agent through a single run to a Result.

    ``max_turns`` is the run-control knob: the maximum number of model calls
    before the Runtime gives up and reports ``Blocked``. A plain int — not a
    separate ``StopPolicy`` object — until a second stop dimension actually
    appears (avoiding speculative generality).
    """

    max_turns: int = 8

    async def run(self, agent: Agent, prompt: str) -> Result:
        """Drive ``agent`` from an initial user ``prompt`` to a terminal Result."""
        tools_by_name = {tool.spec.name: tool for tool in agent.tools}
        context = Context(
            messages=(Message(role="user", content=(TextBlock(prompt),)),),
            turn=0,
        )
        while True:
            if context.turn >= self.max_turns:
                return Blocked(
                    reason=f"exceeded max_turns ({self.max_turns})",
                    context=context,
                )
            response = await agent.model.complete(
                system=agent.instructions,
                messages=context.messages,
                tools=tuple(tool.spec for tool in agent.tools),
            )
            context = replace(
                context,
                messages=(*context.messages, response.message),
                turn=context.turn + 1,
            )

            calls = [
                block
                for block in response.message.content
                if isinstance(block, ToolUseBlock)
            ]
            if response.stop_reason != "tool_use" or not calls:
                return Completed(
                    output=_final_text(response.message),
                    context=context,
                )

            result_blocks, blocked = await self._dispatch(
                calls, tools_by_name, agent.permissions
            )
            if blocked is not None:
                return Blocked(reason=blocked, context=context)
            context = replace(
                context,
                messages=(
                    *context.messages,
                    Message(role="user", content=tuple(result_blocks)),
                ),
            )

    async def _dispatch(
        self,
        calls: list[ToolUseBlock],
        tools_by_name: dict[str, Tool],
        permissions: Permissions,
    ) -> tuple[list[ToolResultBlock], str | None]:
        """Run each requested tool call, enforcing permissions.

        Returns the tool-result blocks to feed back to the model, plus an optional
        block reason. A denied permission or an unknown tool ends the run
        (``Blocked``); an *error raised by the tool itself* is reported back to the
        model as an error result so it can recover, rather than aborting the run.
        """
        result_blocks: list[ToolResultBlock] = []
        for call in calls:
            if not _is_permitted(permissions, call.name):
                return result_blocks, f"permission denied for tool {call.name!r}"
            tool = tools_by_name.get(call.name)
            if tool is None:
                return result_blocks, f"unknown tool {call.name!r}"
            outcome = await tool(call.input)
            result_blocks.append(
                ToolResultBlock(
                    tool_use_id=call.id,
                    content=outcome.content,
                    is_error=outcome.is_error,
                )
            )
        return result_blocks, None
