"""The console's concrete agents: a conversational orchestrator and a file-reading
planner specialist, plus the ``plan_file`` tool that dispatches the planner and
lands its output as a proposed artifact.

These are *application* choices — a "plan" is a domain concept — which is why they
live in the console layer, not in the generic harness.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique.code import Coordinator
from agentique.core import Agent, Model
from agentique.core.tool import ToolResult, ToolSpec
from agentique.tools import AskHuman, ReadFile

_PLANNER_INSTRUCTIONS = (
    "You are a planning specialist. Read the file you are given with the read_file "
    "tool, then reply with a concise, actionable, numbered plan based on its "
    "contents. Do not ask questions; produce the plan directly."
)

_ORCHESTRATOR_INSTRUCTIONS = (
    "You are the orchestrator of a small coding-assistant console, talking with a "
    "human operator.\n"
    "- When the operator wants a plan for a file, call the plan_file tool with the "
    "file path. It runs a planner sub-agent that reads the file and stores a "
    "proposed plan artifact; tell the operator the artifact id so they can approve "
    "or reject it.\n"
    "- To get the operator's next message or decision, call the ask_human tool with "
    "your question. Always end your turn by calling ask_human (by itself) so the "
    "operator can reply — never finish silently.\n"
    "- Keep replies short."
)


class PlanFile:
    """Dispatch the planner sub-agent on a file; land its plan as an artifact.

    Synchronous agents-as-tools: it runs the planner via the Coordinator (so the
    plan is stored as a ``proposed`` artifact and the run is a tracked Session),
    then returns an acknowledgement — the artifact id and a preview — into the
    orchestrator's turn.
    """

    def __init__(self, planner: Agent, coordinator: Coordinator) -> None:
        self._planner = planner
        self._coordinator = coordinator

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="plan_file",
            description=(
                "Dispatch a planner sub-agent to read a file and produce a plan, "
                "stored as a proposed artifact for the operator to approve."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file to plan around.",
                    }
                },
                "required": ["path"],
            },
        )

    async def __call__(self, arguments: Mapping[str, object]) -> ToolResult:
        path = arguments.get("path")
        if not isinstance(path, str):
            return ToolResult(content="argument 'path' must be a string", is_error=True)
        prompt = f"Read the file at {path!r} and produce a concise plan."
        session = await self._coordinator.dispatch(self._planner, prompt, kind="plan")
        if session.state != "done" or session.artifact is None:
            detail = session.error or session.state
            return ToolResult(
                content=f"the planner did not produce a plan ({detail})",
                is_error=True,
            )
        artifact = session.artifact
        return ToolResult(
            content=(
                f"Stored a proposed plan as artifact {artifact.id} (from {path}). "
                f"The operator can approve or reject it.\n\n{artifact.payload[:240]}"
            )
        )


def build_planner(model: Model) -> Agent:
    """A specialist that reads a file and returns a plan."""
    return Agent(
        name="planner",
        instructions=_PLANNER_INSTRUCTIONS,
        model=model,
        tools=(ReadFile(),),
    )


def build_orchestrator(model: Model, coordinator: Coordinator, planner: Agent) -> Agent:
    """The conversational orchestrator: talks to the operator, dispatches the
    planner via ``plan_file``, and pauses for input via ``ask_human``."""
    return Agent(
        name="orchestrator",
        instructions=_ORCHESTRATOR_INSTRUCTIONS,
        model=model,
        tools=(AskHuman(), PlanFile(planner, coordinator)),
    )
