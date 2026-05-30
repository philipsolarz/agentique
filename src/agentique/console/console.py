"""Console: the operator-facing application object.

Holds one Coordinator and a conversational orchestrator agent, owns the
conversation (the orchestrator's run, threaded across turns via the pause/resume
spine), and exposes the operator's verbs: send a message, and approve or reject a
stored artifact. The REPL in :mod:`agentique.console.cli` is a thin shell over it.

The orchestrator yields for operator input by calling ``ask_human``; each operator
message resumes that paused run, so the conversation Context is threaded across
turns without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentique.code import Artifact, Coordinator
from agentique.console.agents import build_orchestrator, build_planner
from agentique.core import (
    Agent,
    Blocked,
    Completed,
    Context,
    Model,
    NeedsHuman,
    Paused,
    Result,
    Runtime,
    TextBlock,
)


@dataclass(frozen=True, slots=True)
class Turn:
    """What to show the operator after one message.

    ``awaiting_input`` is True when the orchestrator paused for the operator's
    reply; ``done`` is True when the conversation ended (the run completed or was
    blocked).
    """

    message: str
    awaiting_input: bool
    done: bool


def _assistant_text(context: Context) -> str:
    """The text the orchestrator spoke in its most recent assistant turn."""
    for message in reversed(context.messages):
        if message.role == "assistant":
            return "".join(b.text for b in message.content if isinstance(b, TextBlock))
    return ""


class Console:
    """Drives one orchestrator conversation and the shared artifact lifecycle."""

    def __init__(
        self,
        orchestrator: Agent,
        *,
        coordinator: Coordinator | None = None,
        runtime: Runtime | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._coordinator = coordinator if coordinator is not None else Coordinator()
        self._runtime = runtime if runtime is not None else Runtime()
        self._paused: Paused | None = None
        self._done = False
        self._last_result: Result | None = None

    @property
    def coordinator(self) -> Coordinator:
        return self._coordinator

    @property
    def done(self) -> bool:
        return self._done

    @property
    def last_result(self) -> Result | None:
        """The core Result of the most recent send/resume — for capture/inspection."""
        return self._last_result

    async def send(self, text: str) -> Turn:
        """Send the operator's ``text`` to the orchestrator and surface its reply."""
        if self._done:
            raise RuntimeError("the conversation has ended")
        if self._paused is None:
            result = await self._runtime.run(self._orchestrator, text)
        else:
            result = await self._runtime.resume(self._orchestrator, self._paused, text)
        return self._handle(result)

    def _handle(self, result: Result) -> Turn:
        self._last_result = result
        match result:
            case NeedsHuman(question=question, paused=paused):
                self._paused = paused
                said = _assistant_text(paused.context)
                message = f"{said}\n\n{question}".strip() if said else question
                return Turn(message=message, awaiting_input=True, done=False)
            case Completed(output=output):
                self._paused = None
                self._done = True
                return Turn(message=output, awaiting_input=False, done=True)
            case Blocked(reason=reason):
                self._paused = None
                self._done = True
                return Turn(
                    message=f"[blocked] {reason}", awaiting_input=False, done=True
                )

    async def approve(self, artifact_id: str) -> Artifact:
        """Mark a stored artifact approved."""
        return await self._coordinator.approve_artifact(artifact_id)

    async def reject(self, artifact_id: str) -> Artifact:
        """Mark a stored artifact rejected."""
        return await self._coordinator.reject_artifact(artifact_id)

    async def artifacts(self) -> tuple[Artifact, ...]:
        """All artifacts that have converged in the shared store."""
        return await self._coordinator.store.artifacts()


def build_console(model: Model, *, planner_model: Model | None = None) -> Console:
    """Wire a Console with one orchestrator and a file-reading planner.

    ``planner_model`` defaults to ``model``; pass a separate one (e.g. a distinct
    ``StubModel``) to script the orchestrator and planner independently in tests.
    """
    coordinator = Coordinator()
    planner = build_planner(planner_model if planner_model is not None else model)
    orchestrator = build_orchestrator(model, coordinator, planner)
    return Console(orchestrator, coordinator=coordinator)
