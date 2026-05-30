"""A minimal inline Model stub for Runtime tests.

Kept private to agentique-core's own test suite so core depends on no sibling
package — not even at dev time. (The richer, shipped ``StubModel`` lives in
``agentique-testing`` and is tested there.) This stub satisfies the ``Model``
protocol structurally and replays scripted responses in order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agentique.core.messages import (
    Message,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolUseBlock,
)
from agentique.core.tool import ToolSpec


class StubModel:
    """Replays the given responses, one per ``complete`` call."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    @staticmethod
    def text(text: str, stop_reason: StopReason = "end_turn") -> ModelResponse:
        """Build a single text response — a convenience for scripting."""
        return ModelResponse(
            message=Message(role="assistant", content=(TextBlock(text),)),
            stop_reason=stop_reason,
        )

    @staticmethod
    def tool_call(
        tool_id: str, name: str, arguments: Mapping[str, object]
    ) -> ModelResponse:
        """Build a single tool-use response — a convenience for scripting."""
        return ModelResponse(
            message=Message(
                role="assistant",
                content=(ToolUseBlock(id=tool_id, name=name, input=arguments),),
            ),
            stop_reason="tool_use",
        )

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        response = self._responses[self._index]
        self._index += 1
        return response
