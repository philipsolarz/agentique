"""The Dispatch tool: the orchestrator's generic, transparent way to put a
specialist to work.

This is the agents-as-tools primitive — the same shape as a coding assistant
dispatching a sub-agent. Calling it runs a named :class:`~agentique.role.Role`
to completion via the Coordinator's explicit ``dispatch_role`` capability, which
tracks the run as a Session and lands the specialist's output as a *proposed*
artifact in the shared store. It then returns the artifact id and a preview into
the orchestrator's turn so it can narrate what happened.

Dispatch is therefore a **visible** coordination act, not a hidden side effect: it
produces a tracked Session and an operator-visible proposed artifact. Crucially it
never *promotes* anything — moving an artifact to ``approved``/``rejected`` stays
the operator's separate, explicit act (``/approve``, ``/reject``). This is the
generic replacement for the old domain-baked ``plan_file`` tool.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique import Coordinator
from agentique.core.tool import ToolResult, ToolSpec


class Dispatch:
    """Dispatch a specialist role on a task; land its output as a proposed artifact."""

    def __init__(self, coordinator: Coordinator) -> None:
        self._coordinator = coordinator

    def _role_names(self) -> str:
        return ", ".join(role.name for role in self._coordinator.roles())

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="dispatch",
            description=(
                "Dispatch a specialist to work on a task. Runs the named role to "
                "completion and stores its output as a proposed artifact for the "
                f"operator to approve or reject. Available roles: {self._role_names()}."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": "The specialist role to dispatch.",
                    },
                    "task": {
                        "type": "string",
                        "description": "The task and any context for the specialist.",
                    },
                },
                "required": ["role", "task"],
            },
        )

    async def __call__(self, arguments: Mapping[str, object]) -> ToolResult:
        role = arguments.get("role")
        task = arguments.get("task")
        if not isinstance(role, str):
            return ToolResult(content="argument 'role' must be a string", is_error=True)
        if not isinstance(task, str):
            return ToolResult(content="argument 'task' must be a string", is_error=True)
        if self._coordinator.role(role) is None:
            return ToolResult(
                content=f"unknown role {role!r} (available: {self._role_names()})",
                is_error=True,
            )
        session = await self._coordinator.dispatch_role(role, task)
        if session.state != "done" or session.artifact is None:
            detail = session.error or session.state
            return ToolResult(
                content=f"the {role} did not finish ({detail})", is_error=True
            )
        artifact = session.artifact
        return ToolResult(
            content=(
                f"{role} produced a proposed {artifact.kind} artifact "
                f"{artifact.id}. The operator can /approve or /reject it.\n\n"
                f"{artifact.payload[:400]}"
            )
        )
