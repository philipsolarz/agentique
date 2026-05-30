"""A generic multi-agent tool: spawn another agent as a granted capability.

``Delegate`` wraps a *child* :class:`~agentique.core.agent.Agent` and exposes it
as a :class:`~agentique.core.tool.Tool`. When the parent's model calls it, the
Delegate runs the child to completion on its own Runtime and returns the child's
output as the tool result.

This is deliberately *mechanism, not policy*. It carries **no** orchestration
logic — no routing, planning, retries, or topology rules. Because delegation is
just a granted tool, multi-agent topology (hub-and-spoke, nested chains) is a
function of *which* Delegate tools you grant to *which* agents, decided by the
caller (and, later, the application layer) — never baked in here. It is gated by
the parent's permissions exactly like any other tool.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique.core.agent import Agent
from agentique.core.result import Blocked, Completed, NeedsHuman
from agentique.core.runtime import Runtime
from agentique.core.tool import ToolResult, ToolSpec


class Delegate:
    """Expose a child agent as a tool the parent agent may invoke.

    ``name`` is how the parent's model addresses the delegation (so one parent can
    hold several Delegates to different children under distinct names). The child
    is driven by ``runtime`` — its own loop, independent of the parent's.
    """

    def __init__(
        self,
        child: Agent,
        *,
        name: str,
        description: str,
        runtime: Runtime | None = None,
    ) -> None:
        self._child = child
        self._name = name
        self._description = description
        self._runtime = runtime if runtime is not None else Runtime()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=self._description,
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task to hand to the delegated agent.",
                    }
                },
                "required": ["prompt"],
            },
        )

    async def __call__(self, arguments: Mapping[str, object]) -> ToolResult:
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str):
            return ToolResult(
                content="argument 'prompt' must be a string", is_error=True
            )
        result = await self._runtime.run(self._child, prompt=prompt)
        match result:
            case Completed(output=output):
                return ToolResult(content=output)
            case NeedsHuman(question=question):
                # Surface the child's pause as an error result: the generic tool
                # boundary is a single string, so the parent's model is told the
                # child needs input rather than the run silently stalling. Richer
                # human-in-the-loop propagation is an application-layer concern.
                return ToolResult(
                    content=f"delegated agent needs human input: {question}",
                    is_error=True,
                )
            case Blocked(reason=reason):
                return ToolResult(
                    content=f"delegated agent blocked: {reason}", is_error=True
                )
