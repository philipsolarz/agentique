"""Pluggable agent routing strategies.

``AgentRouter`` is the main registry. Routing logic is delegated to a
pluggable strategy. The default strategy is ``LLMRouter``, which uses
``ctx.sample()`` to implement a **Plan → Execute → Verify** pattern:

* **Plan** — ``LLMRouter.aselect()`` builds a rich capability manifest
  from the full agent registry (name, description, skills, endpoint) and
  calls ``ctx.sample()`` with ``result_type=RoutingDecision`` so the
  client LLM produces a **validated, structured** routing decision —
  including a confidence score, reasoning, and ordered fallback agents.
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
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentique.core.errors import AgentNotFoundError
from agentique.core.types import AgentInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured routing decision
# ---------------------------------------------------------------------------


class RoutingDecision(BaseModel):
    """Structured routing decision produced by the LLM plan phase.

    When the MCP client supports structured sampling,
    ``LLMRouter.aselect()`` requests a ``RoutingDecision`` via
    ``ctx.sample(result_type=RoutingDecision)``. This gives the gateway:

    * A **validated** agent selection (not raw string matching).
    * A **confidence score** for observability and threshold-based alerts.
    * A **reasoning** string for audit logs and span attributes.
    * An ordered **fallback list** for graceful degradation when the
      primary agent is unavailable or returns an error.
    * A **decomposition hint** for future multi-step orchestration.
    """

    agent_id: str = Field(
        description=(
            "Exact name of the selected agent, copied verbatim from the "
            "available-agents list."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Routing confidence: 0.0 = very uncertain, 1.0 = near-certain. "
            "Use values below 0.5 to flag ambiguous requests."
        ),
    )
    reasoning: str = Field(
        description=(
            "One-sentence explanation of why this agent was chosen. "
            "Used in audit logs and OpenTelemetry span attributes."
        )
    )
    fallback_agents: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of fallback agent names to try if the primary "
            "agent fails or is unavailable. May be empty."
        ),
    )
    requires_decomposition: bool = Field(
        default=False,
        description=(
            "Hint: True when the request clearly spans multiple distinct "
            "domains that no single agent can fully handle. Signals that "
            "future multi-step orchestration would benefit this request."
        ),
    )


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

_DEFAULT_SYSTEM_PROMPT = (
    "You are an intelligent agent routing assistant. "
    "Given a user request and a numbered list of available agents with their "
    "descriptions and skills, select the single best agent to handle the request.\n\n"
    "Produce a structured routing decision with:\n"
    "  • agent_id   — the exact agent name from the list\n"
    "  • confidence — a float from 0.0 (very uncertain) to 1.0 (near-certain)\n"
    "  • reasoning  — one concise sentence explaining your choice\n"
    "  • fallback_agents — ordered list of backup agent names (may be empty)\n"
    "  • requires_decomposition — true only when the request clearly spans "
    "multiple distinct domains that no single agent can fully satisfy\n\n"
    "Use high confidence (≥0.8) only when the match is unambiguous. "
    "Always set agent_id to the exact agent name as it appears in the list."
)

_DEFAULT_VERIFY_PROMPT = (
    "You are a quality-assurance assistant. Given the original user request "
    "and an agent's response, determine whether the response adequately "
    "answers the request. Reply with exactly 'YES' if satisfied, or 'NO' "
    "followed by a brief reason."
)


class LLMRouter:
    """Routes using the MCP client's LLM via ``ctx.sample()``.

    Implements a **Plan → Execute → Verify** pattern:

    *Plan* — ``aselect()`` builds a rich capability manifest (name,
    description, skills, endpoint) and requests a structured
    ``RoutingDecision`` from the client LLM via
    ``ctx.sample(result_type=RoutingDecision)``. When the client does not
    support structured sampling, it falls back to plain-text sampling and
    wraps the response in a minimal ``RoutingDecision``.

    The ``RoutingDecision`` includes a **fallback agent cascade**: if the
    primary ``agent_id`` cannot be matched in the registry, ``aselect()``
    tries each entry in ``fallback_agents`` before raising
    ``AgentNotFoundError``.

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
        low_confidence_threshold: Log a warning when the LLM's confidence
            is below this value. Defaults to ``0.5``.
        sampling_fallback: Optional callable invoked when ``ctx.sample()``
            is unavailable (client does not support sampling). Signature::

                async def fallback(message: str, available: list[AgentInfo]) -> str

            Must return the name of the agent to route to. When ``None``
            (default), a missing sampling capability raises ``RuntimeError``.
        enable_introspection: When ``True``, passes ``inspect_agent`` and
            ``list_agents`` tool callables to ``ctx.sample(tools=[...])``.
            This lets the routing LLM call these tools to query agent details
            before producing a ``RoutingDecision`` (the "first inflection"
            agentic planning loop). Defaults to ``False``.
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        verify_prompt: str | None = None,
        enable_verification: bool = False,
        low_confidence_threshold: float = 0.5,
        sampling_fallback: Callable[..., Any] | None = None,
        enable_introspection: bool = False,
    ) -> None:
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._verify_prompt = verify_prompt or _DEFAULT_VERIFY_PROMPT
        self._enable_verification = enable_verification
        self._low_confidence_threshold = low_confidence_threshold
        self._sampling_fallback = sampling_fallback
        self._enable_introspection = enable_introspection

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
        requests a validated ``RoutingDecision`` from the client LLM via
        ``ctx.sample(result_type=RoutingDecision)``.

        If the client does not support structured sampling (``TypeError`` or
        ``AttributeError``), falls back to plain-text sampling and wraps the
        response in a minimal ``RoutingDecision`` with ``confidence=0.5``.

        After obtaining a ``RoutingDecision``, the method attempts to match
        ``decision.agent_id`` against *available*. On failure it cascades
        through ``decision.fallback_agents`` in order. If all candidates
        fail, ``AgentNotFoundError`` is raised.

        Args:
            message: The user's request.
            available: All currently visible agents with full capabilities.
            ctx: FastMCP ``Context`` exposing ``sample()``.

        Returns:
            The selected ``AgentInfo``.

        Raises:
            RuntimeError: If ``ctx`` is ``None``, lacks sampling support, or
                if the LLM sampling call itself fails.
            AgentNotFoundError: If no suggested agent exists in *available*.
        """
        # When sampling is unavailable, use the registered fallback function.
        if ctx is None or not hasattr(ctx, "sample"):
            if self._sampling_fallback is not None:
                agent_name = await _call_fallback(
                    self._sampling_fallback, message, available
                )
                return _match_agent(agent_name, available)
            raise RuntimeError(
                "LLMRouter requires a FastMCP Context with sampling support. "
                "Ensure the MCP client advertises the 'sampling' capability, "
                "or pass sampling_fallback= to LLMRouter()."
            )

        manifest = _build_agent_manifest(available)
        available_names = [a.name for a in available]
        plan_prompt = (
            f"User request:\n{message}\n\n"
            f"Available agents:\n{manifest}\n\n"
            f"Select the single best agent. "
            f"Agent names must be exactly from: {', '.join(available_names)}"
        )

        # Build optional introspection tools for the agentic planning loop.
        introspection_tools = (
            _make_introspection_tools(available) if self._enable_introspection else None
        )

        decision: RoutingDecision | None = None

        # Attempt structured sampling — result_type=RoutingDecision gives a
        # validated Pydantic object directly.
        try:
            sample_kwargs: dict[str, Any] = dict(
                system_prompt=self._system_prompt,
                result_type=RoutingDecision,
            )
            if introspection_tools:
                sample_kwargs["tools"] = introspection_tools
            result = await ctx.sample(plan_prompt, **sample_kwargs)
            raw = result.result if hasattr(result, "result") else result
            if isinstance(raw, RoutingDecision):
                decision = raw
            elif isinstance(raw, dict):
                decision = RoutingDecision.model_validate(raw)
        except (TypeError, AttributeError):
            # Client does not support structured sampling — fall through.
            pass
        except Exception as exc:
            logger.debug(
                "Structured sampling failed (%s); falling back to plain text", exc
            )

        # Plain-text sampling fallback.
        if decision is None:
            try:
                plain_kwargs: dict[str, Any] = dict(system_prompt=self._system_prompt)
                if introspection_tools:
                    plain_kwargs["tools"] = introspection_tools
                result = await ctx.sample(
                    plan_prompt
                    + "\n\nReply with ONLY the agent name — no explanation.",
                    **plain_kwargs,
                )
                text = str(
                    result.text if hasattr(result, "text") else result
                ).strip()
                decision = RoutingDecision(
                    agent_id=text,
                    confidence=0.5,
                    reasoning="Plain-text routing (structured sampling unavailable)",
                    fallback_agents=[],
                    requires_decomposition=False,
                )
            except Exception as exc:
                # If ctx.sample() itself is broken, try the user-supplied fallback.
                if self._sampling_fallback is not None:
                    logger.warning(
                        "ctx.sample() failed (%s); using sampling_fallback", exc
                    )
                    try:
                        agent_name = await _call_fallback(
                            self._sampling_fallback, message, available
                        )
                        return _match_agent(agent_name, available)
                    except Exception as fb_exc:
                        raise RuntimeError(
                            f"LLM routing and sampling fallback both failed: "
                            f"sample={exc!r}, fallback={fb_exc!r}"
                        ) from fb_exc
                raise RuntimeError(
                    f"LLM routing failed: ctx.sample() raised {exc!r}"
                ) from exc

        # Observability: log and annotate the decision.
        _log_routing_decision(decision, self._low_confidence_threshold)
        _annotate_span(decision)

        # Resolve primary agent, then fallbacks.
        candidates = [decision.agent_id, *decision.fallback_agents]
        for candidate in candidates:
            try:
                agent = _match_agent(candidate, available)
                if candidate != decision.agent_id:
                    logger.info(
                        "LLM router: primary '%s' not found; fell back to '%s'",
                        decision.agent_id,
                        candidate,
                    )
                return agent
            except AgentNotFoundError:
                continue

        raise AgentNotFoundError(
            f"LLM routing failed: none of the suggested agents "
            f"({candidates!r}) exist in the registry. "
            f"Available: {available_names}"
        )

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
    """Build a numbered, multi-line capability manifest for the LLM prompt.

    Each entry is formatted as::

        1. agent_name
           Description: <description or 'No description available'>
           Skills: <comma-separated skills or 'general purpose'>
           Endpoint: <base_url or 'local'>
    """
    lines: list[str] = []
    for i, agent in enumerate(agents, 1):
        skills = ", ".join(agent.skills) if agent.skills else "general purpose"
        description = agent.description or "No description available"
        endpoint = agent.base_url or "local"
        lines.append(
            f"{i}. {agent.name}\n"
            f"   Description: {description}\n"
            f"   Skills: {skills}\n"
            f"   Endpoint: {endpoint}"
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


def _log_routing_decision(
    decision: RoutingDecision,
    low_confidence_threshold: float,
) -> None:
    """Emit structured log entries for the routing decision."""
    if decision.confidence < low_confidence_threshold:
        logger.warning(
            "LLM router: low-confidence routing decision "
            "(agent=%r, confidence=%.2f, reasoning=%r)",
            decision.agent_id,
            decision.confidence,
            decision.reasoning,
        )
    elif decision.requires_decomposition:
        logger.info(
            "LLM router: routing to '%s' (confidence=%.2f) — "
            "request may benefit from decomposition: %s",
            decision.agent_id,
            decision.confidence,
            decision.reasoning,
        )
    else:
        logger.info(
            "LLM router: selected '%s' (confidence=%.2f): %s",
            decision.agent_id,
            decision.confidence,
            decision.reasoning,
        )

    if decision.fallback_agents:
        logger.debug(
            "LLM router: fallback chain for '%s': %s",
            decision.agent_id,
            decision.fallback_agents,
        )


async def _call_fallback(
    fallback: Callable[..., Any],
    message: str,
    available: list[AgentInfo],
) -> str:
    """Invoke a sampling fallback function, handling both sync and async callables."""
    import inspect as _inspect

    result = fallback(message, available)
    if _inspect.isawaitable(result):
        result = await result
    return str(result)


def _make_introspection_tools(available: list[AgentInfo]) -> list[Callable[..., Any]]:
    """Create lightweight introspection callables for the agentic planning loop.

    These are passed as ``tools=`` to ``ctx.sample()`` so the routing LLM can
    query agent details before committing to a routing decision — the "first
    inflection" pattern from the Agentique architecture document.

    Two tools are returned:

    ``inspect_agent(name: str) -> dict``
        Returns the description, skills, and endpoint of a named agent.

    ``list_agents() -> list[str]``
        Returns the names of all visible agents.
    """
    # Build a name→AgentInfo map for fast lookup.
    _registry: dict[str, AgentInfo] = {a.name: a for a in available}

    def inspect_agent(name: str) -> dict[str, Any]:
        """Return the description, skills, and endpoint of a named agent.

        Args:
            name: The exact agent name as it appears in the available-agents list.
        """
        info = _registry.get(name)
        if info is None:
            return {"error": f"Agent '{name}' not found", "available": list(_registry)}
        return {
            "name": info.name,
            "description": info.description or "",
            "skills": list(info.skills),
            "endpoint": info.base_url or "local",
        }

    def list_agents() -> list[str]:
        """Return the names of all currently visible agents."""
        return list(_registry)

    return [inspect_agent, list_agents]


def _annotate_span(decision: RoutingDecision) -> None:
    """Add OpenTelemetry span attributes for the routing decision.

    Silently no-ops when OpenTelemetry is not configured or the current
    span is not recording.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("agentique.routing.agent_id", decision.agent_id)
            span.set_attribute(
                "agentique.routing.confidence", decision.confidence
            )
            span.set_attribute("agentique.routing.reasoning", decision.reasoning)
            span.set_attribute(
                "agentique.routing.requires_decomposition",
                decision.requires_decomposition,
            )
            if decision.fallback_agents:
                span.set_attribute(
                    "agentique.routing.fallback_agents",
                    ",".join(decision.fallback_agents),
                )
    except Exception:
        pass
