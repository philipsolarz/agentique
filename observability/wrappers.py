"""Recording wrappers over the ``Model`` and ``Tool`` seams.

These compose against the core Protocols — they do not hook into core. Each wrapper
**observes only**: it returns the inner object's response/result unchanged, so a
wrapped run produces an identical :class:`~agentique.core.result.Result` to an
unwrapped one. The only added effect is one event recorded per call.

Timing is the part that must be exact: ``time.perf_counter`` (monotonic) spans only
the inner ``await`` — started immediately before it and stopped immediately after,
before any event is constructed — so the latency numbers reflect the model/tool,
not the recorder. A raised inner call is the event we most want, so it is recorded
(marked with the exception type) and then re-raised, never swallowed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from agentique.core.messages import (
    ContentBlock,
    Message,
    ModelResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agentique.core.model import Model
from agentique.core.tool import Tool, ToolResult, ToolSpec
from observability.events import ContentBlockSummary, ModelCallEvent, ToolCallEvent
from observability.recorder import Recorder


def _summarize_block(block: ContentBlock) -> ContentBlockSummary:
    """Reduce one returned content block to its kind and a diagnostic char size."""
    match block:
        case TextBlock(text=text):
            return ContentBlockSummary(kind="text", size=len(text))
        case ToolUseBlock(input=input):
            return ContentBlockSummary(kind="tool_use", size=len(repr(dict(input))))
        case ToolResultBlock(content=content):
            return ContentBlockSummary(kind="tool_result", size=len(content))


class RecordingModel:
    """Wraps a ``Model``, recording one :class:`ModelCallEvent` per ``complete``."""

    def __init__(self, inner: Model, recorder: Recorder) -> None:
        self._inner = inner
        self._recorder = recorder

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        system_len = len(system)
        message_count = len(messages)
        tool_names = tuple(tool.name for tool in tools)

        start = time.perf_counter()
        try:
            response = await self._inner.complete(
                system=system, messages=messages, tools=tools
            )
        except Exception as exc:
            latency_s = time.perf_counter() - start
            self._recorder.record(
                ModelCallEvent(
                    system_len=system_len,
                    message_count=message_count,
                    tool_names=tool_names,
                    stop_reason=None,
                    blocks=(),
                    latency_s=latency_s,
                    raised=type(exc).__name__,
                )
            )
            raise
        latency_s = time.perf_counter() - start

        self._recorder.record(
            ModelCallEvent(
                system_len=system_len,
                message_count=message_count,
                tool_names=tool_names,
                stop_reason=response.stop_reason,
                blocks=tuple(_summarize_block(b) for b in response.message.content),
                latency_s=latency_s,
            )
        )
        return response


class RecordingTool:
    """Wraps a ``Tool``, recording one :class:`ToolCallEvent` per call. ``spec`` is a
    property delegating to the inner tool, mirroring the ``Tool`` protocol."""

    def __init__(self, inner: Tool, recorder: Recorder) -> None:
        self._inner = inner
        self._recorder = recorder

    @property
    def spec(self) -> ToolSpec:
        return self._inner.spec

    async def __call__(self, arguments: Mapping[str, object]) -> ToolResult:
        tool_name = self._inner.spec.name
        # Snapshot the inputs before the call so neither the tool nor the Runtime
        # can change what we recorded by reusing or mutating the mapping afterward.
        arguments_snapshot = dict(arguments)

        start = time.perf_counter()
        try:
            result = await self._inner(arguments)
        except Exception as exc:
            latency_s = time.perf_counter() - start
            self._recorder.record(
                ToolCallEvent(
                    tool_name=tool_name,
                    arguments=arguments_snapshot,
                    result_len=None,
                    is_error=None,
                    latency_s=latency_s,
                    raised=type(exc).__name__,
                )
            )
            raise
        latency_s = time.perf_counter() - start

        self._recorder.record(
            ToolCallEvent(
                tool_name=tool_name,
                arguments=arguments_snapshot,
                result_len=len(result.content),
                is_error=result.is_error,
                latency_s=latency_s,
            )
        )
        return result


if TYPE_CHECKING:
    # Confirm both wrappers satisfy their Protocols, checked by ``ty``.
    def _conformance(inner_model: Model, inner_tool: Tool, recorder: Recorder) -> None:
        _m: Model = RecordingModel(inner_model, recorder)
        _t: Tool = RecordingTool(inner_tool, recorder)
