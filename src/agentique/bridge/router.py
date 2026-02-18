"""Pluggable agent routing strategies.

``AgentRouter`` is the main registry. Routing logic is delegated to a
pluggable strategy. The default strategy is ``LLMRouter``, which uses
``ctx.sample()`` to implement a **Plan → Execute → Verify** pattern:

* **Plan** — ``LLMRouter.aselect()`` builds a rich capability manifest
  from the full agent registry (name, description, skills, endpoint) and
  calls ``ctx.sample()`` to let the client LLM choose the best agent.
* **Verify** — ``LLMRouter.averify()`` optionally calls ``ctx.sample()``
  a second time to validate the agent's response before it is returned.

There are no deterministic keyword-matching fallbacks. All multi-agent
routing decisions are delegated to the client's LLM.

Usage::

    router = AgentRouter(agents)              # defaults to LLMRouter()

    # Inside an async tool:
    agent = await router.aresolve(message=msg, ctx=ctx)

    # Optional post-response quality gate:
    ok = await router.strategy.averify(msg, agent_response, ctx=ctx)
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Protocol, runtime_checkable

from agentique.core.errors import AgentNotFoundError
from agentique.core.types import AgentInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DirectRouter — single-agent or explicit-target setups
# ---------------------------------------------------------------------------


class DirectRouter:
    """Routes to a single named agent (for single-agent setups).

    Falls back to the first registered agent if the target is not found.
    """

    def __init__(self, target: str) -> None:
        self._target = target

    def select(self, message: str, available: list[AgentInfo]) -> AgentInfo:
        for agent in available:
            if agent.name == self._target:
                return agent
        return available[0]


# ---------------------------------------------------------------------------
# LLMRouter — Plan → Execute → Verify
# ---------------------------------------------------------------------------


class LLMRouter:
    """Routes using the MCP client's LLM via ``ctx.sample()``.

    Implements a **Plan → Execute → Verify** pattern:

    *Plan* — ``aselect()`` builds a rich capability manifest (name,
    description, skills, endpoint) and sends it to the client LLM via
    ``ctx.sample()``.

    *Verify* — ``averify()`` optionally calls ``ctx.sample()`` a second
    time to validate that the returned response is satisfactory.

    The synchronous ``select()`` method raises ``RuntimeError`` to force
    callers to the async path. There is no keyword-matching fallback.

    Args:
        system_prompt: Override the system prompt used during agent selection.
        verify_prompt: Override the system prompt used during verification.
        enable_verification: When ``True``, ``averify()`` performs an active
            LLM quality check. Defaults to ``False`` (verification is a
            no-op, always returning ``True``).
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        verify_prompt: str | None = None,
        enable_verification: bool = False,
    ) -> None:
        self._system_prompt = system_prompt or (
            "You are an intelligent agent routing assistant. Given a user request "
            "and a list of agents with their capabilities, select the single best "
            "agent to handle the request. Consider the agent's description, skills, "
            "and specialisation. Reply with ONLY the agent name — no explanation."
        )
        self._verify_prompt = verify_prompt or (
            "You are a quality-assurance assistant. Given the original user request "
            "and an agent's response, determine whether the response adequately "
            "answers the request. Reply with exactly 'YES' if satisfied, or 'NO' "
            "followed by a brief reason."
        )
        self._enable_verification = enable_verification

    # -- sync path (not supported) --

    def select(self, message: str, available: list[AgentInfo]) -> AgentInfo:
        """Synchronous selection is not supported by LLMRouter.

        Raises:
            RuntimeError: Always. Call ``AgentRouter.aresolve()`` instead.
        """
        raise RuntimeError(
            "LLMRouter requires an async context with ctx.sample() support. "
            "Use AgentRouter.aresolve(message=..., ctx=ctx) instead of resolve()."
        )

    # -- async PLAN phase --

    async def aselect(
        self,
        message: str,
        available: list[AgentInfo],
        ctx: Any = None,
    ) -> AgentInfo:
        """PLAN phase: select the best agent via LLM sampling.

        Builds a structured capability manifest from all visible agents and
        calls ``ctx.sample()`` with that manifest so the client LLM can make
        an informed routing decision.

        Args:
            message: The user's request.
            available: All currently visible agents with full capabilities.
            ctx: FastMCP ``Context`` exposing ``sample()``.

        Returns:
            The selected ``AgentInfo``.

        Raises:
            RuntimeError: If ``ctx`` is ``None`` or lacks sampling support.
            AgentNotFoundError: If the LLM names an agent not in *available*.
        """
        if ctx is None or not hasattr(ctx, "sample"):
            raise RuntimeError(
                "LLMRouter requires a FastMCP Context with sampling support. "
                "Ensure the MCP client advertises the 'sampling' capability."
            )

        manifest = _build_agent_manifest(available)
        agent_names = [a.name for a in available]
        plan_prompt = (
            f"User request:\n{message}\n\n"
            f"Available agents (name | description | skills | endpoint):\n"
            f"{manifest}\n\n"
            f"Select the single best agent. Reply with exactly one of: "
            f"{', '.join(agent_names)}"
        )

        chosen_name: str | None = None
        try:
            # Structured sampling — result_type constrains to valid agent names
            result = await ctx.sample(
                plan_prompt,
                system_prompt=self._system_prompt,
                result_type=agent_names,
            )
            chosen_name = str(
                result.result if hasattr(result, "result") else result
            ).strip()

        except (TypeError, AttributeError):
            # Plain-text sampling fallback
            result = await ctx.sample(
                plan_prompt,
                system_prompt=self._system_prompt,
            )
            chosen_name = str(
                result.text if hasattr(result, "text") else result
            ).strip()

        agent = _match_agent(chosen_name, available)
        logger.info("LLM router (plan) selected agent '%s'", agent.name)
        return agent

    # -- async VERIFY phase --

    async def averify(
        self,
        original_message: str,
        agent_response: str,
        *,
        ctx: Any = None,
    ) -> bool:
        """VERIFY phase: validate the agent's response via LLM sampling.

        Calls ``ctx.sample()`` to ask the client LLM whether *agent_response*
        adequately addresses *original_message*. Returns ``True`` when
        verification is disabled (default) or when ``ctx`` is unavailable
        (fail-open behaviour).

        Args:
            original_message: The original user request.
            agent_response: The text the agent returned.
            ctx: FastMCP ``Context`` exposing ``sample()``.

        Returns:
            ``True`` if the response is adequate, ``False`` otherwise.
        """
        if not self._enable_verification:
            return True

        if ctx is None or not hasattr(ctx, "sample"):
            logger.debug("LLM verification skipped: no ctx.sample available")
            return True

        verify_prompt = (
            f"Original request:\n{original_message}\n\n"
            f"Agent response:\n{agent_response}\n\n"
            "Is this response adequate? Reply YES or NO."
        )
        try:
            result = await ctx.sample(
                verify_prompt,
                system_prompt=self._verify_prompt,
            )
            text = str(
                result.text if hasattr(result, "text") else result
            ).strip().upper()
            satisfied = text.startswith("YES")
            if not satisfied:
                logger.info("LLM verification REJECTED response: %.120s", text)
            return satisfied
        except Exception as exc:
            logger.warning("LLM verification failed (%s); treating as OK", exc)
            return True


# ---------------------------------------------------------------------------
# AgentRouter
# ---------------------------------------------------------------------------


class AgentRouter:
    """Main agent registry and router.

    Maintains a registry of known agents and delegates async selection to a
    pluggable ``RoutingStrategy``. Defaults to ``LLMRouter`` for fully
    LLM-driven routing decisions.
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
        self._strategy: RoutingStrategy = strategy or LLMRouter()

        if agents:
            for agent in agents:
                self.register(agent)

        if default is not None:
            self._default = default

    @property
    def strategy(self) -> RoutingStrategy:
        """The active routing strategy."""
        return self._strategy

    def register(self, agent: AgentInfo) -> None:
        self._agents[agent.name] = agent
        if self._default is None:
            self._default = agent.name

    def unregister(self, name: str) -> bool:
        """Remove an agent from the registry. Returns ``True`` if found."""
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
        """Resolve an agent by explicit name, skill, or single-agent default.

        This synchronous path only handles direct lookups. Multi-agent
        routing by message content requires the async ``aresolve()`` method.

        Raises:
            RuntimeError: When multiple agents are registered and no
                ``name`` or ``skill`` is given (use ``aresolve()``).
        """
        if name:
            return self.describe(name)

        if skill:
            for agent in self._agents.values():
                if skill in agent.skills:
                    return agent

        # Single-agent shortcut — no routing decision needed
        if len(self._agents) == 1:
            return next(iter(self._agents.values()))

        if message and len(self._agents) > 1:
            raise RuntimeError(
                "Multi-agent routing by message content requires an LLM context. "
                "Use AgentRouter.aresolve(message=..., ctx=ctx) instead."
            )

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
        """Async resolve — uses LLM-based routing via ``ctx.sample()``.

        For single-agent setups or explicit name/skill targets, no LLM call
        is made. LLM routing is only invoked when multiple agents are
        registered and no explicit target is provided.
        """
        if name:
            return self.describe(name)

        if skill:
            for agent in self._agents.values():
                if skill in agent.skills:
                    return agent

        # Single-agent shortcut
        if len(self._agents) == 1:
            return next(iter(self._agents.values()))

        if message and hasattr(self._strategy, "aselect"):
            return await self._strategy.aselect(
                message, list(self._agents.values()), ctx=ctx,
            )

        if self._default is None:
            raise RuntimeError("No agents registered in the router.")
        return self.describe(self._default)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_agent_manifest(agents: list[AgentInfo]) -> str:
    """Build a tabular capability manifest for the LLM routing prompt."""
    lines: list[str] = []
    for agent in agents:
        skills = ", ".join(agent.skills) if agent.skills else "none"
        description = agent.description or "No description"
        endpoint = agent.base_url or "local"
        lines.append(
            f"• {agent.name} | {description} | skills: {skills} | endpoint: {endpoint}"
        )
    return "\n".join(lines)


def _match_agent(chosen_name: str, available: list[AgentInfo]) -> AgentInfo:
    """Match a chosen agent name string to an ``AgentInfo`` in *available*.

    Performs exact match first, then case-insensitive, then substring.
    """
    lower = chosen_name.strip().lower()

    for agent in available:
        if agent.name == chosen_name.strip():
            return agent

    for agent in available:
        if agent.name.lower() == lower:
            return agent

    for agent in available:
        if agent.name.lower() in lower or lower in agent.name.lower():
            return agent

    raise AgentNotFoundError(
        f"LLM selected unknown agent '{chosen_name}'. "
        f"Available: {[a.name for a in available]}"
    )
