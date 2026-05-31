"""Context compaction: keep a run inside a token budget at the pre-model point.

A :class:`Compactor` proposes a trimmed :class:`~agentique.core.context.Context`
before each model call; :class:`CompactionMiddleware` runs it at the pre-model
interposition point and emits a :class:`~agentique.core.events.Compaction` event.
The one shipped rung, :class:`EvictOldestToolResults`, replaces the *oldest*
tool-result payloads with a short placeholder until the context fits — no model
call, no judgement, a pure function of the context (so it stays deterministic).
Tool-result blocks are blanked, never removed, so each ``tool_use`` keeps its
matching ``tool_result``.

Size is a cheap character count standing in for tokens; a real tokenizer is
headroom (add a Compactor that uses one — the seam does not change). Model-summary
compaction, summary memory, and skill re-injection are likewise additional
Compactor rungs, not built here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from agentique.core.context import Context
from agentique.core.events import Compaction, EventSink, NullSink
from agentique.core.messages import (
    Message,
    OpaqueBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agentique.core.middleware import CallNext, Point, PreModelPoint


class Compactor(Protocol):
    """Proposes a Context that fits ``budget``. Pure: no model call, no I/O."""

    def compact(self, context: Context, budget: int) -> Context: ...


def estimate_size(context: Context) -> int:
    """A cheap character-count proxy for the context's token footprint."""
    total = 0
    for message in context.messages:
        for block in message.content:
            match block:
                case TextBlock(text=text):
                    total += len(text)
                case ToolResultBlock(content=content):
                    total += len(content)
                case ToolUseBlock(input=input):
                    total += len(repr(dict(input)))
                case OpaqueBlock(provider_data=provider_data):
                    total += len(repr(dict(provider_data)))
    return total


@dataclass(frozen=True, slots=True)
class EvictOldestToolResults:
    """Evict the oldest tool-result payloads (blanking them to ``placeholder``)
    until the context fits the budget. Tool results carry the bulk of a run's
    tokens and age out of relevance first, so they are the cheapest thing to drop.
    """

    placeholder: str = "[evicted to fit the context budget]"

    def compact(self, context: Context, budget: int) -> Context:
        size = estimate_size(context)
        if size <= budget:
            return context
        rows = [list(message.content) for message in context.messages]
        for row in rows:
            for index, block in enumerate(row):
                if size <= budget:
                    break
                if (
                    isinstance(block, ToolResultBlock)
                    and block.content != self.placeholder
                ):
                    saved = len(block.content) - len(self.placeholder)
                    row[index] = ToolResultBlock(
                        tool_use_id=block.tool_use_id,
                        content=self.placeholder,
                        is_error=block.is_error,
                    )
                    if saved > 0:
                        size -= saved
            if size <= budget:
                break
        new_messages = tuple(
            Message(role=message.role, content=tuple(rows[index]))
            for index, message in enumerate(context.messages)
        )
        return replace(context, messages=new_messages)


def _evicted_count(before: Context, after: Context) -> int:
    """How many tool-result blocks were blanked between ``before`` and ``after``.

    Compaction only rewrites block *content* (never adds/removes blocks), so the
    two contexts line up position-for-position.
    """
    count = 0
    for mb, ma in zip(before.messages, after.messages, strict=False):
        for bb, ba in zip(mb.content, ma.content, strict=False):
            if (
                isinstance(bb, ToolResultBlock)
                and isinstance(ba, ToolResultBlock)
                and bb.content != ba.content
            ):
                count += 1
    return count


class CompactionMiddleware:
    """Runs a :class:`Compactor` at the pre-model point and emits the event.

    Opt-in: add it to an Engine's middleware to bound that run's context. It
    proposes the compacted Context via the return path (the onion's contract), so
    the model call — and the conversation going forward — uses the trimmed history.
    """

    def __init__(
        self,
        compactor: Compactor,
        *,
        budget: int,
        sink: EventSink | None = None,
    ) -> None:
        self._compactor = compactor
        self._budget = budget
        self._sink: EventSink = sink if sink is not None else NullSink()

    async def handle(self, point: Point, call_next: CallNext) -> Any:
        if isinstance(point, PreModelPoint):
            proposed = await call_next()
            if isinstance(proposed, Context):
                compacted = self._compactor.compact(proposed, self._budget)
                evicted = _evicted_count(proposed, compacted)
                if evicted:
                    self._sink.emit(Compaction(evicted_blocks=evicted))
                return compacted
            return proposed
        return await call_next()
