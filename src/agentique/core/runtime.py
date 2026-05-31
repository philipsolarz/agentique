"""The Engine: drives a declarative Agent to a Result, through the middleware onion.

The Agent declares *what* it is (role, model, tools, permissions); the Engine owns
*how a run proceeds* — assemble the conversation, call the model, dispatch any tool
calls, decide when the run is done, emit a Result. Run-control (the turn limit and
the stop decision) lives here, not on the Agent, per the agreed Agent/Engine split.
The Engine knows nothing about *other* agents; coordinating several is the
Scheduler's job (a peer in the core), which uses an Engine to advance each one.

Every step runs through the middleware onion (:mod:`agentique.core.middleware`) at
a fixed set of points — turn, pre-model, model-call, tool-call. With the default
empty chain the onion is a pass-through, so the loop below behaves exactly as the
single-agent loop always has; built-in middlewares (tracing, permissions,
compaction) layer on without changing it.

The loop:

1. Guard the turn limit; ``Blocked`` if exceeded.
2. Run one turn (``_one_turn``): compact, call the model, and — if the model
   requested tools (``stop_reason.kind == "tool_use"``) — dispatch them and fold
   the results back as a user message, continuing; otherwise return a terminal
   Result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

from agentique.core.agent import Agent
from agentique.core.context import Context
from agentique.core.control import PauseRequested, PermissionDenied
from agentique.core.messages import (
    Message,
    ModelResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agentique.core.middleware import (
    Middleware,
    ModelPoint,
    PermissionMiddleware,
    PreModelPoint,
    ToolPoint,
    TurnPoint,
    apply,
)
from agentique.core.result import Blocked, Completed, NeedsHuman, Paused, Result
from agentique.core.run_context import RunContext
from agentique.core.tool import Tool, ToolResult
from agentique.core.validation import validate_output, validate_tool_args


def _final_text(message: Message) -> str:
    """Join the text blocks of an assistant message into the run's output."""
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))


def _terminal(response: ModelResponse, context: Context) -> Result:
    """Map a non-tool-dispatching model response to its terminal Result.

    ``done``/``length``/``refusal`` are genuine terminal completions that carry
    the assistant's text (a refusal *is* output, not an error). ``tool_use``
    reaches here only when the model claimed tools but emitted no tool block, so
    it completes on whatever text it produced. ``paused`` (a provider mid-turn
    pause we do not continue) and ``other`` (a reason the core does not model) are
    surfaced as ``Blocked`` — never folded into a quiet ``Completed``, which is
    the silent-fold defect this replaces.
    """
    kind = response.stop_reason.kind
    if kind in ("done", "length", "refusal", "tool_use"):
        return Completed(output=_final_text(response.message), context=context)
    return Blocked(
        reason=f"unhandled stop_reason {kind!r} (raw {response.stop_reason.raw!r})",
        context=context,
    )


async def _just(value: Any) -> Any:
    """A trivial coroutine returning ``value`` — the core action at a pass-through
    point (e.g. pre-model with no compactor proposes the unchanged Context)."""
    return value


@dataclass(frozen=True, slots=True)
class Engine:
    """Drives one Agent through a single run to a Result.

    ``max_turns`` is the run-control knob: the maximum number of model calls before
    the Engine gives up and reports ``Blocked``. ``middleware`` is the onion applied
    at every interposition point; the default empty chain is a pass-through.
    """

    max_turns: int = 8
    middleware: tuple[Middleware, ...] = ()

    async def run(
        self, agent: Agent, prompt: str, *, ctx: RunContext | None = None
    ) -> Result:
        """Drive ``agent`` from an initial user ``prompt`` to a terminal Result.

        ``ctx`` is the run handle passed to tools (run id, event emit, dispatch); a
        Scheduler supplies one bound to the run, and a bare Engine defaults to a
        capability-free handle (no dispatch).
        """
        context = Context(
            messages=(Message(role="user", content=(TextBlock(prompt),)),),
            turn=0,
        )
        return await self._drive(
            agent, context, ctx if ctx is not None else RunContext()
        )

    async def resume(
        self,
        agent: Agent,
        paused: Paused,
        answer: str,
        *,
        ctx: RunContext | None = None,
    ) -> Result:
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
        return await self._drive(
            agent, context, ctx if ctx is not None else RunContext()
        )

    async def _drive(self, agent: Agent, context: Context, ctx: RunContext) -> Result:
        """The loop shared by ``run`` and ``resume``: turn after turn until terminal.

        Each turn runs through the turn-point of the onion; a turn either proposes a
        fresh Context to continue from or returns a terminal Result.
        """
        tools_by_name = {tool.spec.name: tool for tool in agent.tools}
        # The effective chain enforces the agent's permissions around any
        # caller-supplied middleware, so permission policy is always applied while
        # still using the single onion mechanism (it is a no-op at non-tool points).
        chain = (PermissionMiddleware(agent.permissions), *self.middleware)
        while True:
            if context.turn >= self.max_turns:
                return Blocked(
                    reason=f"exceeded max_turns ({self.max_turns})",
                    context=context,
                )
            current = context
            outcome = await apply(
                chain,
                TurnPoint(current),
                lambda current=current: self._one_turn(
                    agent, current, tools_by_name, chain, ctx
                ),
            )
            if isinstance(outcome, Context):
                context = outcome
                continue
            return cast(Result, outcome)

    async def _one_turn(
        self,
        agent: Agent,
        context: Context,
        tools_by_name: dict[str, Tool],
        chain: tuple[Middleware, ...],
        ctx: RunContext,
    ) -> Context | Result:
        """Run one turn: compact, call the model, dispatch any tools.

        Returns a fresh Context to continue from, or a terminal Result.
        """
        # Pre-model point: a compactor may propose a trimmed Context here.
        context = cast(
            Context,
            await apply(
                chain,
                PreModelPoint(context),
                lambda: _just(context),
            ),
        )
        tool_specs = tuple(tool.spec for tool in agent.tools)
        # Model-call point.
        response = cast(
            ModelResponse,
            await apply(
                chain,
                ModelPoint(context, tool_specs),
                lambda: agent.model.complete(
                    system=agent.instructions,
                    messages=context.messages,
                    tools=tool_specs,
                ),
            ),
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
        if response.stop_reason.kind != "tool_use" or not calls:
            terminal = _terminal(response, context)
            # Typed-output gate: a Completed whose text fails the declared
            # output_type is not terminal — feed the validation error back and let
            # the model self-correct on the next turn (bounded by max_turns).
            if isinstance(terminal, Completed) and agent.output_type is not None:
                error = validate_output(agent.output_type, terminal.output)
                if error is not None:
                    return replace(
                        context,
                        messages=(
                            *context.messages,
                            Message(role="user", content=(TextBlock(error),)),
                        ),
                    )
            return terminal

        try:
            result_blocks, blocked = await self._dispatch(
                calls, tools_by_name, chain, ctx
            )
        except PauseRequested as pause:
            # A tool — ``ask_human`` or a permission ``ask`` — asked to pause for
            # human input. It must be the sole call in its turn so exactly one
            # tool_use is left unanswered — the pairing ``resume`` answers. Reject
            # the mixed turn loudly rather than building a half-answered conversation.
            if len(calls) != 1:
                return Blocked(
                    reason="ask_human must be the sole tool call in its turn",
                    context=context,
                )
            return NeedsHuman(
                question=pause.question,
                paused=Paused(context=context, pending_tool_use_id=calls[0].id),
            )
        except PermissionDenied as denied:
            # A ``deny`` rule matched: end the run, not recoverable by the model.
            return Blocked(reason=denied.reason, context=context)
        if blocked is not None:
            return Blocked(reason=blocked, context=context)
        return replace(
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
        chain: tuple[Middleware, ...],
        ctx: RunContext,
    ) -> tuple[list[ToolResultBlock], str | None]:
        """Run each requested tool call through the tool-call onion.

        Returns the tool-result blocks to feed back to the model, plus an optional
        block reason. An unknown tool ends the run (``Blocked``). Permission policy
        is enforced by the permission middleware at the tool-call point (a ``deny``
        raises ``PermissionDenied``, an ``ask`` raises ``PauseRequested``); a tool
        that *raises* otherwise is folded into an error result the model can recover
        from.
        """
        result_blocks: list[ToolResultBlock] = []
        for call in calls:
            tool = tools_by_name.get(call.name)
            if tool is None:
                return result_blocks, f"unknown tool {call.name!r}"
            outcome = await self._invoke_tool(tool, call, chain, ctx)
            result_blocks.append(
                ToolResultBlock(
                    tool_use_id=call.id,
                    content=outcome.content,
                    is_error=outcome.is_error,
                )
            )
        return result_blocks, None

    async def _invoke_tool(
        self,
        tool: Tool,
        call: ToolUseBlock,
        chain: tuple[Middleware, ...],
        ctx: RunContext,
    ) -> ToolResult:
        """Invoke one tool through the tool-call point.

        The onion runs first (so permission ``deny``/``ask`` pre-empt the call);
        the core then validates typed arguments and runs the tool with the run
        handle ``ctx``, folding a raised exception into an error result
        (``PauseRequested`` excepted — it propagates).
        """

        async def _core() -> ToolResult:
            args_model = tool.spec.args_model
            if args_model is not None:
                error = validate_tool_args(args_model, call.input)
                if error is not None:
                    # Typed-args gate: a mismatch becomes a self-correctable error
                    # result; the tool is not run with invalid arguments.
                    return ToolResult(content=error, is_error=True)
            try:
                return await tool(ctx, call.input)
            except PauseRequested:
                raise
            except Exception as exc:
                return ToolResult(
                    content=f"tool {call.name!r} raised: {exc!r}", is_error=True
                )

        return cast(ToolResult, await apply(chain, ToolPoint(call), _core))
