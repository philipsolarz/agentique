"""A deterministic, offline ``Tool`` for tests.

``EchoTool`` echoes one of its arguments back as the result and records every
call it received, so tests can assert both that the Runtime dispatched the call
and what it fed back — without crossing any real external boundary. It is shipped
(not test-only) so the application layer can reuse it, mirroring :class:`StubModel`.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique.core.run_context import RunContext
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

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(
            content=str(arguments.get("value", "")), is_error=self._is_error
        )
