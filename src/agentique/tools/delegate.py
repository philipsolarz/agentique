"""A generic multi-agent tool: dispatch another registered agent as a capability.

``Delegate`` exposes a *child* agent — registered on the Scheduler under
``child_id`` — as a :class:`~agentique.core.tool.Tool`. When the parent's model
calls it, the Delegate dispatches the child through ``ctx.dispatch`` (so the child
runs as its own Scheduler run, recording the parent's run as its parent) and maps
the child's real structured Result back:

* ``Completed`` → the child's output, folded into the parent conversation.
* ``NeedsHuman`` → re-raised as a real pause of the *parent* run (the child's
  question reaches the human through the same pause spine), not flattened to an
  error string.
* ``Blocked`` → an error result the parent's model can react to.

This is deliberately *mechanism, not policy*: it carries no routing/planning logic.
Multi-agent topology is a function of which Delegate tools are granted to which
agents, and it is permission-gated like any other tool. Under a bare Engine (no
Scheduler, so ``ctx.dispatch`` is ``None``) it reports a clear error rather than
spawning.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique.core.control import PauseRequested
from agentique.core.result import Blocked, Completed, NeedsHuman
from agentique.core.run_context import RunContext
from agentique.core.tool import ToolResult, ToolSpec


class Delegate:
    """Expose a registered child agent as a tool the parent agent may invoke.

    ``name`` is how the parent's model addresses the delegation (so one parent can
    hold several Delegates to different children under distinct names); ``child_id``
    is the id the child agent is registered under on the Scheduler.
    """

    def __init__(self, child_id: str, *, name: str, description: str) -> None:
        self._child_id = child_id
        self._name = name
        self._description = description

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

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str):
            return ToolResult(
                content="argument 'prompt' must be a string", is_error=True
            )
        if ctx.dispatch is None:
            return ToolResult(content="dispatch requires a scheduler", is_error=True)
        result = await ctx.dispatch(self._child_id, prompt)
        match result:
            case Completed(output=output):
                return ToolResult(content=output)
            case NeedsHuman(question=question):
                # Propagate the child's pause as a real pause of the parent run,
                # rather than flattening it to an error string.
                raise PauseRequested(question)
            case Blocked(reason=reason):
                return ToolResult(
                    content=f"delegated agent blocked: {reason}", is_error=True
                )
