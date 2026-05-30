"""The Model seam: the LLM client boundary.

A direct Anthropic Messages API client implements this in A2, alongside a
deterministic ``StubModel`` for offline, reproducible tests. The seam is
structural (Protocol), so either — and any future provider — satisfies it by
shape rather than by inheritance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agentique.core.messages import Message, ModelResponse
from agentique.core.tool import ToolSpec


class Model(Protocol):
    """An LLM that, given a system prompt, a conversation, and the tools it is
    allowed to call, produces the next assistant response."""

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        """Produce the next assistant message for the given conversation state."""
        ...
