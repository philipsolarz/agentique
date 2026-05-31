"""A deterministic, offline Model for tests.

``StubModel`` returns a pre-scripted sequence of responses and records every
call it received. This lets the Agent/Engine loop (A3) be exercised
deterministically without a network or API key — the central reason the Model
seam exists. It is shipped (not test-only) so the application layer can reuse it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from agentique.core.messages import (
    Message,
    ModelResponse,
    StopKind,
    StopReason,
    TextBlock,
    ToolUseBlock,
)
from agentique.core.tool import ToolSpec


class StubModelExhausted(RuntimeError):
    """Raised when a :class:`StubModel` is called more times than it has scripted
    responses. A dedicated type (not ``StopIteration``, which PEP 479 turns into
    a ``RuntimeError`` when raised inside a coroutine) so callers can catch the
    condition precisely."""


@dataclass(frozen=True, slots=True)
class StubCall:
    """A snapshot of the arguments one ``complete`` call received, for
    assertions in tests."""

    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]


class StubModel:
    """A Model that replays ``responses`` in order, one per ``complete`` call.

    Not frozen and not a dataclass: it holds mutable call-recording state. It
    satisfies the :class:`~agentique.core.model.Model` protocol structurally.
    """

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses: list[ModelResponse] = list(responses)
        self._index = 0
        self.calls: list[StubCall] = []

    @staticmethod
    def text(
        text: str, *, kind: StopKind = "done", raw: str | None = None
    ) -> ModelResponse:
        """Build a single text response — a convenience for scripting runs.

        Defaults to a normal ``done`` completion; pass ``kind`` (and optionally a
        ``raw`` vendor string) to script a ``length``/``refusal``/``other`` stop.
        """
        return ModelResponse(
            message=Message(role="assistant", content=(TextBlock(text),)),
            stop_reason=StopReason(kind=kind, raw=raw if raw is not None else kind),
        )

    @staticmethod
    def tool_call(
        tool_id: str, name: str, arguments: Mapping[str, object]
    ) -> ModelResponse:
        """Build a single tool-use response — a convenience for scripting runs."""
        return ModelResponse(
            message=Message(
                role="assistant",
                content=(ToolUseBlock(id=tool_id, name=name, input=arguments),),
            ),
            stop_reason=StopReason(kind="tool_use", raw="tool_use"),
        )

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        self.calls.append(
            StubCall(system=system, messages=tuple(messages), tools=tuple(tools))
        )
        if self._index >= len(self._responses):
            raise StubModelExhausted(
                f"StubModel exhausted after {len(self._responses)} response(s)"
            )
        response = self._responses[self._index]
        self._index += 1
        return response
