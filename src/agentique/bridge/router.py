"""Pluggable agent routing strategies.

``AgentRouter`` is the main registry. Routing logic is swappable via
strategy classes: ``KeywordRouter`` (default), ``DirectRouter`` (single
agent), ``LLMRouter`` (uses ``ctx.sample()``), and
``WeightedKeywordRouter`` (keyword matching with scoring).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Protocol, runtime_checkable

from agentique.core.errors import AgentNotFoundError
from agentique.core.types import AgentInfo

logger = logging.getLogger(__name__)


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


class WeightedKeywordRouter:
    """Enhanced keyword router with configurable skill weights.

    Allows assigning different weights to different skills for
    finer-grained routing control::

        router = WeightedKeywordRouter(
            weights={"calculator": 2.0, "text": 1.0}
        )
    """

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        description_weight: float = 0.5,
    ) -> None:
        self._weights = weights or {}
        self._desc_weight = description_weight

    def select(self, message: str, available: list[AgentInfo]) -> AgentInfo:
        msg_lower = message.lower()
        best: AgentInfo | None = None
        best_score = 0.0

        for agent in available:
            score = 0.0
            for skill in agent.skills:
                if skill.lower() in msg_lower:
                    score += self._weights.get(skill, 1.0)

            if agent.description:
                desc_words = agent.description.lower().split()[:8]
                matches = sum(1 for w in desc_words if w in msg_lower)
                score += matches * self._desc_weight

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


class LLMRouter:
    """Routes by asking the MCP client's LLM via ``ctx.sample()``.

    This strategy leverages FastMCP 3.0's sampling capability to let
    the MCP client's own LLM decide which agent is best suited for a
    request. Falls back to ``KeywordRouter`` if sampling is unavailable.

    Usage::

        router = AgentRouter(agents, strategy=LLMRouter())

    Note: This router requires a FastMCP ``Context`` object to be
    available. When used outside of a tool handler (e.g., in tests),
    it falls back to keyword routing.
    """

    def __init__(
        self,
        *,
        fallback: RoutingStrategy | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._fallback = fallback or KeywordRouter()
        self._system_prompt = system_prompt or (
            "You are a routing assistant. Given a user message and a list of "
            "available agents, respond with ONLY the name of the best agent. "
            "Do not explain your reasoning."
        )

    def select(self, message: str, available: list[AgentInfo]) -> AgentInfo:
        """Synchronous fallback — LLM routing requires async context."""
        return self._fallback.select(message, available)

    async def aselect(
        self,
        message: str,
        available: list[AgentInfo],
        ctx: Any = None,
    ) -> AgentInfo:
        """Async routing via LLM sampling.

        Uses ``ctx.sample()`` with ``result_type`` for structured agent
        selection when available, falling back to plain-text sampling.

        Args:
            message: The user's message.
            available: List of available agents.
            ctx: FastMCP Context with ``sample()`` capability.

        Returns:
            The selected agent.
        """
        if ctx is None or not hasattr(ctx, "sample"):
            return self._fallback.select(message, available)

        agent_names = [a.name for a in available]
        agent_descriptions = "\n".join(
            f"- {a.name}: {a.description or 'No description'} "
            f"(skills: {', '.join(a.skills) or 'none'})"
            for a in available
        )

        prompt = (
            f"User message: {message}\n\n"
            f"Available agents:\n{agent_descriptions}\n\n"
            f"Which agent should handle this? Reply with the agent name only."
        )

        try:
            # Try structured sampling with result_type (list of agent names)
            result = await ctx.sample(
                prompt,
                system_prompt=self._system_prompt,
                result_type=agent_names,
            )
            # result_type=list[str] returns the selected agent name
            chosen_name = str(result.result if hasattr(result, "result") else result).strip()

            for agent in available:
                if agent.name.lower() == chosen_name.lower():
                    logger.info(
                        "LLM router (structured) selected agent '%s'",
                        agent.name,
                    )
                    return agent

        except (TypeError, AttributeError):
            # Structured sampling not supported — fall back to plain text
            try:
                result = await ctx.sample(
                    prompt,
                    system_prompt=self._system_prompt,
                )
                chosen_name = str(
                    result.text if hasattr(result, "text") else result
                ).strip().lower()

                for agent in available:
                    if agent.name.lower() == chosen_name:
                        logger.info(
                            "LLM router selected agent '%s' for message",
                            agent.name,
                        )
                        return agent

                # Fuzzy match: check if the LLM included extra text
                for agent in available:
                    if agent.name.lower() in chosen_name:
                        logger.info(
                            "LLM router fuzzy-matched agent '%s'",
                            agent.name,
                        )
                        return agent

                logger.warning(
                    "LLM router returned unknown agent '%s', falling back",
                    chosen_name,
                )
            except Exception as exc:
                logger.warning(
                    "LLM routing failed (%s), falling back to keyword router",
                    exc,
                )

        except Exception as exc:
            logger.warning(
                "LLM routing failed (%s), falling back to keyword router",
                exc,
            )

        return self._fallback.select(message, available)


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

    def unregister(self, name: str) -> bool:
        """Remove an agent from the registry. Returns True if found."""
        if name in self._agents:
            del self._agents[name]
            if self._default == name:
                self._default = next(iter(self._agents), None)
            return True
        return False

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

    async def aresolve(
        self,
        *,
        name: str | None = None,
        skill: str | None = None,
        message: str | None = None,
        ctx: Any = None,
    ) -> AgentInfo:
        """Async resolve — supports LLM-based routing via ``ctx.sample()``.

        Falls back to synchronous ``resolve()`` for non-LLM strategies.
        """
        if name:
            return self.describe(name)

        if skill:
            for agent in self._agents.values():
                if skill in agent.skills:
                    return agent

        # Try async selection if strategy supports it
        if message and len(self._agents) > 1:
            if hasattr(self._strategy, "aselect"):
                return await self._strategy.aselect(
                    message, list(self._agents.values()), ctx=ctx,
                )
            return self._strategy.select(message, list(self._agents.values()))

        if self._default is None:
            raise RuntimeError("No agents registered in the router.")
        return self.describe(self._default)
