"""The Agent: a declarative actor.

An Agent is a *value*, not a runner. It declares a role/system prompt, one
Model, the Tools it may use, and its permissions. It does not drive
itself — the Engine does. Keeping the Agent purely declarative means the same spec
can be inspected, serialized, forked, or handed to different Engines.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from agentique.core.model import Model
from agentique.core.tool import Tool

type Effect = Literal["deny", "ask", "allow"]
"""What a permission rule does to a matching tool call: refuse the run, pause for
human approval, or permit it."""

type ArgPredicate = Callable[[Mapping[str, object]], bool]
"""An optional extra match condition on a tool call's arguments."""

# deny outranks ask outranks allow, so a single deny anywhere among the matching
# rules settles it regardless of order.
_PRECEDENCE: dict[Effect, int] = {"deny": 3, "ask": 2, "allow": 1}


@dataclass(frozen=True, slots=True)
class Rule:
    """One permission rule: an ``effect`` for tool calls it matches.

    ``tool_name`` ``None`` matches any tool; ``arg_predicate`` ``None`` matches any
    arguments. The rule is policy *data* — the specific rules (e.g. "ask before
    write_file") are supplied by higher layers; the mechanism is the core's.
    """

    effect: Effect
    tool_name: str | None = None
    arg_predicate: ArgPredicate | None = None

    def matches(self, tool_name: str, args: Mapping[str, object]) -> bool:
        if self.tool_name is not None and self.tool_name != tool_name:
            return False
        return self.arg_predicate is None or self.arg_predicate(args)


@dataclass(frozen=True, slots=True)
class Permissions:
    """Ordered permission rules resolved with ``deny > ask > allow`` precedence.

    Among the rules matching a tool call, the highest-precedence effect wins (a
    single ``deny`` settles it); if none match, ``default`` applies. The default
    ``Permissions()`` permits everything, preserving "an agent may use every tool
    it holds". Enforcement lives in the built-in permission middleware, not here —
    this is just the policy data. No rule DSL: rules are plain values.
    """

    rules: tuple[Rule, ...] = ()
    default: Effect = "allow"

    def decide(self, tool_name: str, args: Mapping[str, object]) -> Effect:
        """The effect for a tool call: the strongest matching rule, else default."""
        matched = [r.effect for r in self.rules if r.matches(tool_name, args)]
        if not matched:
            return self.default
        return max(matched, key=_PRECEDENCE.__getitem__)

    @classmethod
    def allow_all(cls) -> Permissions:
        """Permit every tool — the default."""
        return cls()

    @classmethod
    def allowlist(cls, names: Iterable[str]) -> Permissions:
        """Permit only the named tools; deny everything else. ``names`` is any
        iterable of tool names (an empty one denies all)."""
        rules = tuple(Rule(effect="allow", tool_name=n) for n in names)
        return cls(rules=rules, default="deny")


@dataclass(frozen=True, slots=True)
class Agent:
    """A declarative agent specification.

    ``output_type``, when set, is a Pydantic model the engine validates the agent's
    final answer against before completing: a malformed answer is fed back for the
    model to self-correct rather than completing the run with invalid output. Left
    ``None`` for agents whose output is free-form text. It is a live class (never
    serialized), so ``Agent`` stays a plain dataclass holding live seam objects.
    """

    name: str
    instructions: str
    model: Model
    tools: tuple[Tool, ...] = ()
    permissions: Permissions = field(default_factory=Permissions)
    output_type: type[BaseModel] | None = None
