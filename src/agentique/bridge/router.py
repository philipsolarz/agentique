"""Pluggable agent routing strategies.

``AgentRouter`` is the main registry. Routing logic is swappable via
strategy classes: ``KeywordRouter`` (default), ``DirectRouter`` (single
agent), and future ``LLMRouter`` (uses ``ctx.sample()``).
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from agentique.core.errors import AgentNotFoundError
from agentique.core.types import AgentInfo


@runtime_checkable
class RoutingStrategy(Protocol):
    """Protocol for routing strategy implementations."""

    def select(
        self,
        message: str,
        available: list[AgentInfo],
    ) -> AgentInfo:
        """Select the best agent for *message*."""
        ...


class KeywordRouter:
    """Routes based on keyword matching against agent skills."""

    def select(self, message: str, available: list[AgentInfo]) -> AgentInfo:
        msg_lower = message.lower()
        best: AgentInfo | None = None
        best_score = 0
        for agent in available:
            score = sum(1 for skill in agent.skills if skill.lower() in msg_lower)
            if agent.description and any(
                w in msg_lower for w in (agent.description.lower().split()[:5])
            ):
                score += 1
            if score > best_score:
                best = agent
                best_score = score
        return best or available[0]


class DirectRouter:
    """Routes to a single named agent (for single-agent setups)."""

    def __init__(self, target: str) -> None:
        self._target = target

    def select(self, message: str, available: list[AgentInfo]) -> AgentInfo:
        for agent in available:
            if agent.name == self._target:
                return agent
        return available[0]


class AgentRouter:
    """Main agent registry and router.

    Maintains a registry of known agents and delegates selection to a
    pluggable ``RoutingStrategy``.
    """

    def __init__(
        self,
        agents: Iterable[AgentInfo] | None = None,
        *,
        default: str | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> None:
        self._agents: dict[str, AgentInfo] = {}
        self._default: str | None = None
        self._strategy = strategy or KeywordRouter()

        if agents:
            for agent in agents:
                self.register(agent)

        if default is not None:
            self._default = default

    def register(self, agent: AgentInfo) -> None:
        self._agents[agent.name] = agent
        if self._default is None:
            self._default = agent.name

    def list_agents(self) -> list[AgentInfo]:
        return list(self._agents.values())

    def describe(self, name: str) -> AgentInfo:
        try:
            return self._agents[name]
        except KeyError:
            raise AgentNotFoundError(
                f"Unknown agent '{name}'. Available: {sorted(self._agents)}"
            )

    def resolve(
        self,
        *,
        name: str | None = None,
        skill: str | None = None,
        message: str | None = None,
    ) -> AgentInfo:
        """Resolve an agent by name, skill, message content, or default."""
        if name:
            return self.describe(name)

        if skill:
            for agent in self._agents.values():
                if skill in agent.skills:
                    return agent

        if message and len(self._agents) > 1:
            return self._strategy.select(message, list(self._agents.values()))

        if self._default is None:
            raise RuntimeError("No agents registered in the router.")
        return self.describe(self._default)
