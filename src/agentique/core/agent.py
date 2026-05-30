"""The Agent: a declarative actor.

An Agent is a *value*, not a runner. It declares a role/system prompt, one
Model, the Tools and Skills it may use, and its permissions. It does not drive
itself — the Runtime (A3) does. Keeping the Agent purely declarative means the
same spec can be inspected, serialized, forked, or handed to different Runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentique.core.model import Model
from agentique.core.skill import Skill
from agentique.core.tool import Tool


@dataclass(frozen=True, slots=True)
class Permissions:
    """Which tools the Runtime may dispatch for this agent.

    Provisional shape: the permission *check* and any approval semantics are
    built and refined at A4. For now this is plain data — an allowlist of tool
    names, where ``None`` means "every tool the agent holds is permitted". Kept
    as a field on the Agent (not extracted into a separate ``Policy`` object) per
    the agreed deferral.
    """

    allowed_tools: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class Agent:
    """A declarative agent specification."""

    name: str
    instructions: str
    model: Model
    tools: tuple[Tool, ...] = ()
    # ``Skill[Any, Any]``: one agent holds skills with differing In/Out types.
    # Python has no existential type to express "a collection of skills, each of
    # some type", so ``Any`` here is the honest representation of a heterogeneous
    # collection — not an escape hatch.
    skills: tuple[Skill[Any, Any], ...] = ()
    permissions: Permissions = field(default_factory=Permissions)
