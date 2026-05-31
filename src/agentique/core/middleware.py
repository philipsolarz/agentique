"""The middleware onion: one ordered async chain around a fixed set of points.

A single mechanism unifies what would otherwise be separate hook systems
(permissions, tracing, compaction). Each middleware is an ``async def handle(self,
point, call_next)`` — ASGI/Starlette style — free to run code before and after the
inner action and to short-circuit by returning without calling ``call_next``. The
chain runs outer→inner inbound and unwinds inner→outer outbound. The Engine
applies the chain at a small, fixed set of interposition points; this is
deliberately *not* "wrap anything".

State flows by **proposal through the return path**, never by mutation: a
pre-model middleware returns the :class:`Context` to use; a tool middleware
returns the :class:`~agentique.core.tool.ToolResult`. The default chain is empty,
so an un-instrumented run takes the trivial path with zero overhead. The whole
chain is awaited sequentially on the run's own task — no concurrency — so a
StubModel-driven run stays deterministic.

The points (each carries what a middleware at that point needs):

* :class:`TurnPoint` — wraps one whole turn (model call + tool dispatch).
* :class:`PreModelPoint` — just before the model call; where compaction proposes a
  trimmed Context.
* :class:`ModelPoint` — wraps the model call itself.
* :class:`ToolPoint` — wraps one tool invocation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from agentique.core.agent import Permissions
from agentique.core.context import Context
from agentique.core.control import PauseRequested, PermissionDenied
from agentique.core.events import (
    EventSink,
    ModelCallFinished,
    ModelCallStarted,
    ToolCalled,
    TurnBoundary,
)
from agentique.core.messages import ModelResponse, ToolUseBlock
from agentique.core.tool import ToolResult, ToolSpec


@dataclass(frozen=True, slots=True)
class TurnPoint:
    """One whole turn is about to run; ``context`` is the turn's starting state."""

    context: Context


@dataclass(frozen=True, slots=True)
class PreModelPoint:
    """Just before the model call — the point a Compactor proposes a trimmed
    Context at. ``context`` is the state about to be sent to the model."""

    context: Context


@dataclass(frozen=True, slots=True)
class ModelPoint:
    """The model call itself, with the conversation and tool specs it will see."""

    context: Context
    tool_specs: tuple[ToolSpec, ...]


@dataclass(frozen=True, slots=True)
class ToolPoint:
    """One tool invocation, identified by the model's ``call``."""

    call: ToolUseBlock


type Point = TurnPoint | PreModelPoint | ModelPoint | ToolPoint
"""The closed set of interposition points. A middleware ``match``es to act only on
the points it cares about and forwards the rest with ``await call_next()``."""

type CallNext = Callable[[], Awaitable[Any]]
"""The inner continuation: invoking it runs the rest of the chain plus the core
action and returns that point's natural result (a Context, ModelResponse, or
ToolResult). Not awaiting it short-circuits the inner chain."""


class Middleware(Protocol):
    """One layer of the onion. Implementations act on the points they recognise
    and forward the rest unchanged."""

    async def handle(self, point: Point, call_next: CallNext) -> Any: ...


def _wrap(mw: Middleware, point: Point, call_next: CallNext) -> CallNext:
    async def _next() -> Any:
        return await mw.handle(point, call_next)

    return _next


async def apply(
    middleware: tuple[Middleware, ...], point: Point, core: CallNext
) -> Any:
    """Run ``core`` wrapped by ``middleware`` at ``point``.

    The first middleware in the tuple is the outermost layer. With an empty tuple
    this is exactly ``await core()`` — the trivial path is preserved byte for byte.
    """
    call_next = core
    for mw in reversed(middleware):
        call_next = _wrap(mw, point, call_next)
    return await call_next()


class TracingMiddleware:
    """The built-in observability middleware: emits the core event vocabulary.

    Events are not a parallel system — they are exactly what this middleware emits
    as the chain runs. It observes only: every point is forwarded unchanged, so a
    traced run produces an identical Result to an untraced one.
    """

    def __init__(self, sink: EventSink) -> None:
        self._sink = sink

    async def handle(self, point: Point, call_next: CallNext) -> Any:
        match point:
            case ModelPoint(context=context, tool_specs=tool_specs):
                self._sink.emit(
                    ModelCallStarted(
                        message_count=len(context.messages),
                        tool_count=len(tool_specs),
                    )
                )
                response = await call_next()
                if isinstance(response, ModelResponse):
                    self._sink.emit(
                        ModelCallFinished(
                            stop_reason=response.stop_reason,
                            usage=response.usage,
                            block_count=len(response.message.content),
                        )
                    )
                return response
            case ToolPoint(call=call):
                result = await call_next()
                if isinstance(result, ToolResult):
                    self._sink.emit(
                        ToolCalled(tool_name=call.name, is_error=result.is_error)
                    )
                return result
            case TurnPoint():
                outcome = await call_next()
                if isinstance(outcome, Context):
                    self._sink.emit(TurnBoundary(turn=outcome.turn))
                return outcome
            case PreModelPoint():
                return await call_next()


class PermissionMiddleware:
    """The built-in permissions middleware: enforces an agent's :class:`Permissions`
    at the tool-call point with ``deny > ask > allow`` precedence.

    A ``deny`` raises :class:`PermissionDenied` (the Engine ends the run as
    ``Blocked``); an ``ask`` raises :class:`PauseRequested`, joining the same human
    pause spine ``ask_human`` uses — so the resumed answer is folded as the call's
    result and the run continues. (Approve-then-actually-run the tool is headroom;
    today an ``ask`` asks the human to supply the result, like any paused tool.)
    The Engine applies this layer around every run, seeded from ``agent.permissions``.
    """

    def __init__(self, permissions: Permissions) -> None:
        self._permissions = permissions

    async def handle(self, point: Point, call_next: CallNext) -> Any:
        if isinstance(point, ToolPoint):
            effect = self._permissions.decide(point.call.name, point.call.input)
            if effect == "deny":
                raise PermissionDenied(
                    f"permission denied for tool {point.call.name!r}"
                )
            if effect == "ask":
                raise PauseRequested(
                    f"Approve tool {point.call.name!r} with arguments "
                    f"{dict(point.call.input)!r}?"
                )
        return await call_next()
