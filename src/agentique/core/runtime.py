"""The Runtime: the engine that drives a declarative Agent to a Result.

The Agent declares *what* it is (role, model, tools, permissions); the
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
from agentique.core.control import PauseRequested
from agentique.core.messages import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agentique.core.result import Blocked, Completed, NeedsHuman, Paused, Result
from agentique.core.tool import Tool, ToolResult


def _final_text(message: Message) -> str:
    """Join the text blocks of an assistant message into the run's output.

    A direct two-line fold rather than a shared helper — trivial enough that a
    dedicated abstraction would not earn its keep.
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
        context = Context(
            messages=(Message(role="user", content=(TextBlock(prompt),)),),
            turn=0,
        )
        return await self._drive(agent, context)

    async def resume(self, agent: Agent, paused: Paused, answer: str) -> Result:
        """Continue a specific paused run, folding the human's ``answer`` in.

        ``paused`` is the self-contained snapshot from a prior ``NeedsHuman``; the
        ``answer`` becomes the tool result for the ``ask_human`` call that paused
        the run. It targets exactly the run ``paused`` describes, so several paused
        runs can be resumed independently. The turn counter is *not* reset, so
        ``max_turns`` still bounds the whole run across any number of pauses.

        Idempotent w.r.t. side effects: tools that ran before the pause already
        have their results baked into ``paused.context``; resume only appends the
        answer and continues forward — it never replays a prior tool call.
        """
        context = replace(
            paused.context,
            messages=(
                *paused.context.messages,
                Message(
                    role="user",
                    content=(
                        ToolResultBlock(
                            tool_use_id=paused.pending_tool_use_id,
                            content=answer,
                        ),
                    ),
                ),
            ),
        )
        return await self._drive(agent, context)

    async def _drive(self, agent: Agent, context: Context) -> Result:
        """The loop shared by ``run`` and ``resume``: model, then tools, repeat.

        Threads a fresh immutable ``context`` each step from the given starting
        point until a terminal Result — so ``run`` (fresh context) and ``resume``
        (context + the human's answer) share one source of loop truth.
        """
        tools_by_name = {tool.spec.name: tool for tool in agent.tools}
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

            try:
                result_blocks, blocked = await self._dispatch(
                    calls, tools_by_name, agent.permissions
                )
            except PauseRequested as pause:
                # A cooperating tool asked to pause for human input. ``ask_human``
                # must be the sole call in its turn so exactly one tool_use is left
                # unanswered — the pairing ``resume`` answers. Reject the mixed
                # turn loudly rather than building a half-answered conversation.
                if len(calls) != 1:
                    return Blocked(
                        reason="ask_human must be the sole tool call in its turn",
                        context=context,
                    )
                return NeedsHuman(
                    question=pause.question,
                    paused=Paused(context=context, pending_tool_use_id=calls[0].id),
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
        (``Blocked``). An exception *raised* by a tool is folded into an error
        result so the model can recover, rather than aborting the run — except
        ``PauseRequested``, a cooperating tool's control signal, which propagates
        to the loop to pause the run.
        """
        result_blocks: list[ToolResultBlock] = []
        for call in calls:
            if not _is_permitted(permissions, call.name):
                return result_blocks, f"permission denied for tool {call.name!r}"
            tool = tools_by_name.get(call.name)
            if tool is None:
                return result_blocks, f"unknown tool {call.name!r}"
            try:
                outcome = await tool(call.input)
            except PauseRequested:
                raise
            except Exception as exc:
                outcome = ToolResult(
                    content=f"tool {call.name!r} raised: {exc!r}", is_error=True
                )
            result_blocks.append(
                ToolResultBlock(
                    tool_use_id=call.id,
                    content=outcome.content,
                    is_error=outcome.is_error,
                )
            )
        return result_blocks, None
