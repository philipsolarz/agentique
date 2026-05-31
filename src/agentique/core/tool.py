"""The Tool seam: an action that crosses an external boundary.

A Tool may perform I/O — filesystem, network, spawning another agent — and is
therefore permission-gated (the check lives in the Runtime, not the tool). The
seam is structural: any object exposing a ``spec`` and an async ``__call__`` of
the right shape *is* a Tool, with no inheritance required.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from agentique.core.run_context import RunContext


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """The model-facing declaration of a tool.

    Carries the model-facing ``name``/``description``/``input_schema`` (arbitrary
    JSON Schema, so its values are typed ``object``) plus an optional
    ``args_model``: a Pydantic model the engine validates the model's arguments
    against *before* the tool runs, turning a mismatch into a self-correctable
    error. ``input_schema`` is what the model sees; ``args_model`` is the
    engine-side check — left ``None`` for tools that validate their own inputs.
    A live class, so ``ToolSpec`` stays a plain dataclass (it is never persisted).
    """

    name: str
    description: str
    input_schema: Mapping[str, object]
    args_model: type[BaseModel] | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of running a tool. The Runtime pairs this with the
    originating call id to build a ``ToolResultBlock`` for the next model turn."""

    content: str
    is_error: bool = False


class Tool(Protocol):
    """An external action the model may invoke, subject to the agent's
    permissions (enforced by the Runtime, not by the tool itself)."""

    @property
    def spec(self) -> ToolSpec:
        """The declaration shown to the model."""
        ...

    async def __call__(
        self, ctx: RunContext, arguments: Mapping[str, object]
    ) -> ToolResult:
        """Execute the tool.

        ``ctx`` is the run handle (run id, event emit, dispatch); a tool that needs
        none of it simply ignores it. ``arguments`` are the model-supplied inputs;
        their values are ``object`` because they originate from arbitrary JSON, not
        a statically known shape.
        """
        ...
