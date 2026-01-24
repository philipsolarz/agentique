from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import AgentDescriptor


class AgentRouter:
    """Minimal routing layer that maps a request to a known agent."""

    def __init__(self, agents: Iterable[AgentDescriptor] | None = None, *, default: str | None = None) -> None:
        self._agents: dict[str, AgentDescriptor] = {}
        self._default: str | None = None  # Initialize before calling register
        if agents:
            for agent in agents:
                self.register(agent)
        # Override default if explicitly provided
        if default is not None:
            self._default = default

    def register(self, agent: AgentDescriptor) -> None:
        self._agents[agent.name] = agent
        if self._default is None:
            self._default = agent.name

    def list_agents(self) -> list[AgentDescriptor]:
        return list(self._agents.values())

    def describe(self, name: str) -> AgentDescriptor:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise KeyError(f"Unknown agent '{name}'. Available agents: {sorted(self._agents)}") from exc

    def resolve(self, *, name: str | None = None, skill: str | None = None) -> AgentDescriptor:
        if name:
            return self.describe(name)
        if skill:
            for agent in self._agents.values():
                if skill in agent.skills:
                    return agent
        if self._default is None:
            raise RuntimeError("No agents registered in the router.")
        return self.describe(self._default)


@dataclass(frozen=True)
class RoutedAgent:
    descriptor: AgentDescriptor
    reason: str
