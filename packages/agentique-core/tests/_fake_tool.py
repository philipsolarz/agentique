"""A minimal inline Tool for Runtime dispatch tests, private to core's suite.

Records the arguments it was called with and returns a scripted result, so tests
can assert both that the Runtime dispatched the call and what it fed back. The
real, shipped tools live in ``agentique-tools``; core stays dependency-free.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique.core.tool import ToolResult, ToolSpec


class EchoTool:
    """A structural Tool that echoes one of its arguments back as the result."""

    def __init__(self, name: str = "echo", *, is_error: bool = False) -> None:
        self._name = name
        self._is_error = is_error
        self.calls: list[Mapping[str, object]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description="Echo back the 'value' argument.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        )

    async def __call__(self, arguments: Mapping[str, object]) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(
            content=str(arguments.get("value", "")), is_error=self._is_error
        )
