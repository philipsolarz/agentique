"""The ``ask_human`` tool: pause a run to consult the human operator.

Unlike an ordinary tool it never returns a result on the happy path — it raises
:class:`~agentique.core.control.PauseRequested`, the control signal the Engine
turns into a ``NeedsHuman`` outcome carrying a resumable
:class:`~agentique.core.result.Paused` snapshot. It is still structurally a
:class:`~agentique.core.tool.Tool` (it exposes a ``spec`` and an async
``__call__``); the Engine, not the tool, owns what pausing means.

Call it ALONE in a turn — never alongside other tools. Resume answers the single
pending tool call, so the Engine rejects a turn that mixes ``ask_human`` with
other calls.
"""

from __future__ import annotations

from collections.abc import Mapping

from agentique.core.control import PauseRequested
from agentique.core.run_context import RunContext
from agentique.core.tool import ToolResult, ToolSpec


class AskHuman:
    """Expose human-input requests as a tool the model may call to pause the run."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ask_human",
            description=(
                "Pause and ask the human operator a question. Call this ALONE — "
                "never alongside other tools — and wait for the answer before "
                "taking any further action."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to put to the human operator.",
                    }
                },
                "required": ["question"],
            },
        )

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        question = arguments.get("question")
        if not isinstance(question, str):
            return ToolResult(
                content="argument 'question' must be a string", is_error=True
            )
        raise PauseRequested(question)
