"""The Tool seam: an action that crosses an external boundary.

A Tool may perform I/O — filesystem, network, spawning another agent — and is
therefore permission-gated (the check lives in the Runtime, not the tool). This
is the deliberate contrast with :class:`~agentique.core.skill.Skill`, which is
pure. The seam is structural: any object exposing a ``spec`` and an async
``__call__`` of the right shape *is* a Tool, with no inheritance required.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """The model-facing declaration of a tool.

    Carries no behavior — only what the model is told: the tool's name, purpose,
    and the JSON Schema of its arguments. ``input_schema`` is arbitrary JSON
    Schema, so its values are typed ``object`` rather than ``Any``.
    """

    name: str
    description: str
    input_schema: Mapping[str, object]


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

    async def __call__(self, arguments: Mapping[str, object]) -> ToolResult:
        """Execute the tool.

        ``arguments`` are the model-supplied inputs; their values are ``object``
        because they originate from arbitrary JSON, not a statically known shape.
        """
        ...
