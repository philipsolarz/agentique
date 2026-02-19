"""Tests for agentique.bridge.router.

Covers:
  - RoutingDecision model validation
  - LLMRouter.aselect() — structured sampling, plain-text fallback,
    fallback agent cascade, confidence logging, decomposition hint
  - LLMRouter.averify() — enabled/disabled, yes/no, sampling failure
  - DirectRouter.select()
  - AgentRouter.resolve() / aresolve() — shortcuts and LLM delegation
  - _build_agent_manifest() — format and content
  - _match_agent() — exact, case-insensitive, substring, not-found
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.anyio

from agentique.bridge.router import (
    AgentRouter,
    DirectRouter,
    LLMRouter,
    RoutingDecision,
    _build_agent_manifest,
    _match_agent,
)
from agentique.core.errors import AgentNotFoundError
from agentique.core.types import AgentInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_agent(
    name: str,
    description: str = "",
    skills: tuple[str, ...] = (),
    base_url: str = "http://example.com",
) -> AgentInfo:
    return AgentInfo(
        name=name,
        base_url=base_url,
        description=description,
        skills=skills,
    )


ALPHA = make_agent("alpha", "Alpha agent for math", ("arithmetic", "algebra"))
BETA = make_agent("beta", "Beta agent for text", ("summarisation", "translation"))
GAMMA = make_agent("gamma", "Gamma agent for code", ("python", "rust"))


def make_decision(**kwargs: Any) -> RoutingDecision:
    defaults = dict(
        agent_id="alpha",
        confidence=0.9,
        reasoning="Best match for the task",
        fallback_agents=[],
        requires_decomposition=False,
    )
    defaults.update(kwargs)
    return RoutingDecision(**defaults)


def make_ctx(decision: RoutingDecision | None = None, plain_text: str | None = None):
    """Create a mock FastMCP Context whose sample() returns *decision* or raises."""
    ctx = MagicMock()

    if decision is not None:
        result = MagicMock()
        result.result = decision
        ctx.sample = AsyncMock(return_value=result)
    elif plain_text is not None:
        result = MagicMock()
        result.text = plain_text
        # First call (structured) raises TypeError; second call returns plain text
        ctx.sample = AsyncMock(
            side_effect=[TypeError("no structured sampling"), result]
        )
    else:
        ctx.sample = AsyncMock(side_effect=TypeError("no sampling"))

    return ctx


# ---------------------------------------------------------------------------
# RoutingDecision — model validation
# ---------------------------------------------------------------------------


class TestRoutingDecision:
    def test_valid_full(self):
        d = RoutingDecision(
            agent_id="alpha",
            confidence=0.85,
            reasoning="Strong skill match",
            fallback_agents=["beta", "gamma"],
            requires_decomposition=False,
        )
        assert d.agent_id == "alpha"
        assert d.confidence == 0.85
        assert d.fallback_agents == ["beta", "gamma"]
        assert d.requires_decomposition is False

    def test_defaults(self):
        d = RoutingDecision(
            agent_id="alpha",
            confidence=0.5,
            reasoning="ok",
        )
        assert d.fallback_agents == []
        assert d.requires_decomposition is False

    def test_confidence_at_bounds(self):
        assert RoutingDecision(agent_id="x", confidence=0.0, reasoning="r").confidence == 0.0
        assert RoutingDecision(agent_id="x", confidence=1.0, reasoning="r").confidence == 1.0

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            RoutingDecision(agent_id="x", confidence=-0.1, reasoning="r")

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            RoutingDecision(agent_id="x", confidence=1.01, reasoning="r")

    def test_missing_agent_id_rejected(self):
        with pytest.raises(ValidationError):
            RoutingDecision(confidence=0.9, reasoning="r")  # type: ignore[call-arg]

    def test_missing_reasoning_rejected(self):
        with pytest.raises(ValidationError):
            RoutingDecision(agent_id="x", confidence=0.9)  # type: ignore[call-arg]

    def test_requires_decomposition_true(self):
        d = RoutingDecision(
            agent_id="x",
            confidence=0.4,
            reasoning="spans two domains",
            requires_decomposition=True,
        )
        assert d.requires_decomposition is True


# ---------------------------------------------------------------------------
# LLMRouter.aselect() — structured sampling path
# ---------------------------------------------------------------------------


class TestLLMRouterAselect:
    @pytest.fixture
    def router(self):
        return LLMRouter()

    @pytest.mark.anyio
    async def test_structured_sampling_selects_primary(self, router):
        decision = make_decision(agent_id="alpha", confidence=0.95)
        ctx = make_ctx(decision=decision)
        result = await router.aselect("math question", [ALPHA, BETA], ctx=ctx)
        assert result is ALPHA

    @pytest.mark.anyio
    async def test_structured_sampling_result_is_routingdecision(self, router):
        """When result.result is already a RoutingDecision, use it directly."""
        decision = make_decision(agent_id="beta")
        ctx = make_ctx(decision=decision)
        agent = await router.aselect("text request", [ALPHA, BETA], ctx=ctx)
        assert agent is BETA

    @pytest.mark.anyio
    async def test_plain_text_fallback(self, router):
        """When structured sampling raises TypeError, fall back to plain text."""
        ctx = make_ctx(plain_text="beta")
        agent = await router.aselect("translate this", [ALPHA, BETA, GAMMA], ctx=ctx)
        assert agent is BETA

    @pytest.mark.anyio
    async def test_plain_text_fallback_case_insensitive(self, router):
        ctx = make_ctx(plain_text="ALPHA")
        agent = await router.aselect("something", [ALPHA, BETA], ctx=ctx)
        assert agent is ALPHA

    @pytest.mark.anyio
    async def test_plain_text_fallback_substring(self, router):
        ctx = make_ctx(plain_text="agent gamma is the best")
        agent = await router.aselect("code review", [ALPHA, BETA, GAMMA], ctx=ctx)
        assert agent is GAMMA

    @pytest.mark.anyio
    async def test_no_ctx_raises(self, router):
        with pytest.raises(RuntimeError, match="sampling"):
            await router.aselect("msg", [ALPHA], ctx=None)

    @pytest.mark.anyio
    async def test_ctx_without_sample_raises(self, router):
        ctx = object()  # no .sample attribute
        with pytest.raises(RuntimeError, match="sampling"):
            await router.aselect("msg", [ALPHA], ctx=ctx)

    @pytest.mark.anyio
    async def test_sampling_failure_raises_runtime_error(self, router):
        """When both structured and plain-text sampling fail, raise RuntimeError."""
        ctx = MagicMock()
        ctx.sample = AsyncMock(side_effect=RuntimeError("network error"))
        with pytest.raises(RuntimeError, match="LLM routing failed"):
            await router.aselect("msg", [ALPHA, BETA], ctx=ctx)

    @pytest.mark.anyio
    async def test_unknown_agent_raises(self, router):
        decision = make_decision(agent_id="nonexistent_agent")
        ctx = make_ctx(decision=decision)
        with pytest.raises(AgentNotFoundError):
            await router.aselect("msg", [ALPHA, BETA], ctx=ctx)

    @pytest.mark.anyio
    async def test_fallback_cascade_used_when_primary_missing(self, router):
        """Primary not in registry → cascade to fallback_agents."""
        decision = make_decision(
            agent_id="nonexistent",
            fallback_agents=["gamma", "beta"],
        )
        ctx = make_ctx(decision=decision)
        agent = await router.aselect("msg", [ALPHA, BETA, GAMMA], ctx=ctx)
        assert agent is GAMMA

    @pytest.mark.anyio
    async def test_fallback_cascade_tries_all(self, router):
        """Try every fallback before failing."""
        decision = make_decision(
            agent_id="ghost1",
            fallback_agents=["ghost2", "beta"],
        )
        ctx = make_ctx(decision=decision)
        agent = await router.aselect("msg", [ALPHA, BETA], ctx=ctx)
        assert agent is BETA

    @pytest.mark.anyio
    async def test_all_fallbacks_fail_raises(self, router):
        decision = make_decision(
            agent_id="ghost1",
            fallback_agents=["ghost2", "ghost3"],
        )
        ctx = make_ctx(decision=decision)
        with pytest.raises(AgentNotFoundError, match="none of the suggested agents"):
            await router.aselect("msg", [ALPHA, BETA], ctx=ctx)

    @pytest.mark.anyio
    async def test_low_confidence_logs_warning(self, router, caplog):
        decision = make_decision(agent_id="alpha", confidence=0.3)
        ctx = make_ctx(decision=decision)
        with caplog.at_level(logging.WARNING, logger="agentique.bridge.router"):
            await router.aselect("msg", [ALPHA, BETA], ctx=ctx)
        assert any("low-confidence" in r.message for r in caplog.records)

    @pytest.mark.anyio
    async def test_requires_decomposition_logs_info(self, router, caplog):
        decision = make_decision(
            agent_id="alpha",
            confidence=0.7,
            requires_decomposition=True,
        )
        ctx = make_ctx(decision=decision)
        with caplog.at_level(logging.INFO, logger="agentique.bridge.router"):
            await router.aselect("msg", [ALPHA, BETA], ctx=ctx)
        assert any("decomposition" in r.message for r in caplog.records)

    @pytest.mark.anyio
    async def test_fallback_use_logged_at_info(self, router, caplog):
        decision = make_decision(
            agent_id="ghost",
            fallback_agents=["beta"],
        )
        ctx = make_ctx(decision=decision)
        with caplog.at_level(logging.INFO, logger="agentique.bridge.router"):
            await router.aselect("msg", [ALPHA, BETA], ctx=ctx)
        assert any("fell back" in r.message for r in caplog.records)

    @pytest.mark.anyio
    async def test_otel_span_annotated(self, router):
        """_annotate_span must not raise even when OTel is not configured."""
        decision = make_decision(agent_id="alpha", confidence=0.9)
        ctx = make_ctx(decision=decision)
        # Should not raise even without a real TracerProvider
        await router.aselect("msg", [ALPHA, BETA], ctx=ctx)

    @pytest.mark.anyio
    async def test_dict_result_accepted(self, router):
        """If result.result is a dict, it should be validated into RoutingDecision."""
        raw_dict = {
            "agent_id": "gamma",
            "confidence": 0.88,
            "reasoning": "code expertise",
        }
        result = MagicMock()
        result.result = raw_dict
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=result)
        agent = await router.aselect("write rust code", [ALPHA, BETA, GAMMA], ctx=ctx)
        assert agent is GAMMA


# ---------------------------------------------------------------------------
# LLMRouter.averify()
# ---------------------------------------------------------------------------


class TestLLMRouterAverify:
    @pytest.mark.anyio
    async def test_disabled_always_true(self):
        router = LLMRouter(enable_verification=False)
        ctx = make_ctx(plain_text="NO it's terrible")
        assert await router.averify("q", "answer", ctx=ctx) is True

    @pytest.mark.anyio
    async def test_enabled_yes_returns_true(self):
        router = LLMRouter(enable_verification=True)
        ctx = MagicMock()
        result = MagicMock()
        result.text = "YES looks good"
        ctx.sample = AsyncMock(return_value=result)
        assert await router.averify("q", "a", ctx=ctx) is True

    @pytest.mark.anyio
    async def test_enabled_no_returns_false(self):
        router = LLMRouter(enable_verification=True)
        ctx = MagicMock()
        result = MagicMock()
        result.text = "NO missing key detail"
        ctx.sample = AsyncMock(return_value=result)
        assert await router.averify("q", "a", ctx=ctx) is False

    @pytest.mark.anyio
    async def test_enabled_no_ctx_returns_true(self):
        router = LLMRouter(enable_verification=True)
        assert await router.averify("q", "a", ctx=None) is True

    @pytest.mark.anyio
    async def test_sampling_failure_returns_true(self):
        """Verification failures are fail-open."""
        router = LLMRouter(enable_verification=True)
        ctx = MagicMock()
        ctx.sample = AsyncMock(side_effect=RuntimeError("timeout"))
        assert await router.averify("q", "a", ctx=ctx) is True


# ---------------------------------------------------------------------------
# DirectRouter
# ---------------------------------------------------------------------------


class TestDirectRouter:
    def test_exact_target(self):
        dr = DirectRouter("beta")
        assert dr.select("anything", [ALPHA, BETA, GAMMA]) is BETA

    def test_fallback_to_first(self):
        dr = DirectRouter("nonexistent")
        assert dr.select("anything", [ALPHA, BETA]) is ALPHA


# ---------------------------------------------------------------------------
# AgentRouter
# ---------------------------------------------------------------------------


class TestAgentRouter:
    def test_register_and_list(self):
        ar = AgentRouter()
        ar.register(ALPHA)
        ar.register(BETA)
        assert set(a.name for a in ar.list_agents()) == {"alpha", "beta"}

    def test_unregister(self):
        ar = AgentRouter([ALPHA, BETA])
        assert ar.unregister("alpha") is True
        assert ar.unregister("ghost") is False
        assert [a.name for a in ar.list_agents()] == ["beta"]

    def test_resolve_single_agent(self):
        ar = AgentRouter([ALPHA])
        assert ar.resolve() is ALPHA

    def test_resolve_by_name(self):
        ar = AgentRouter([ALPHA, BETA])
        assert ar.resolve(name="beta") is BETA

    def test_resolve_by_skill(self):
        ar = AgentRouter([ALPHA, BETA])
        assert ar.resolve(skill="summarisation") is BETA

    def test_resolve_multi_agent_message_raises(self):
        ar = AgentRouter([ALPHA, BETA])
        with pytest.raises(RuntimeError, match="aresolve"):
            ar.resolve(message="help me")

    def test_resolve_unknown_name_raises(self):
        ar = AgentRouter([ALPHA])
        with pytest.raises(AgentNotFoundError):
            ar.resolve(name="ghost")

    @pytest.mark.anyio
    async def test_aresolve_single_agent_no_llm(self):
        ar = AgentRouter([ALPHA])
        result = await ar.aresolve(message="anything", ctx=None)
        assert result is ALPHA

    @pytest.mark.anyio
    async def test_aresolve_explicit_name(self):
        ar = AgentRouter([ALPHA, BETA])
        result = await ar.aresolve(name="beta", ctx=None)
        assert result is BETA

    @pytest.mark.anyio
    async def test_aresolve_explicit_skill(self):
        ar = AgentRouter([ALPHA, BETA, GAMMA])
        result = await ar.aresolve(skill="python", ctx=None)
        assert result is GAMMA

    @pytest.mark.anyio
    async def test_aresolve_multi_agent_delegates_to_strategy(self):
        decision = make_decision(agent_id="gamma")
        ctx = make_ctx(decision=decision)
        ar = AgentRouter([ALPHA, BETA, GAMMA])
        result = await ar.aresolve(message="write code", ctx=ctx)
        assert result is GAMMA

    @pytest.mark.anyio
    async def test_aresolve_no_agents_raises(self):
        ar = AgentRouter()
        with pytest.raises(RuntimeError, match="No agents"):
            await ar.aresolve()

    def test_describe_unknown_raises(self):
        ar = AgentRouter([ALPHA])
        with pytest.raises(AgentNotFoundError):
            ar.describe("ghost")

    def test_llm_router_select_raises(self):
        lr = LLMRouter()
        with pytest.raises(RuntimeError, match="async"):
            lr.select("msg", [ALPHA])


# ---------------------------------------------------------------------------
# _build_agent_manifest
# ---------------------------------------------------------------------------


class TestBuildAgentManifest:
    def test_numbered_entries(self):
        manifest = _build_agent_manifest([ALPHA, BETA])
        assert manifest.startswith("1. alpha")
        assert "2. beta" in manifest

    def test_includes_description(self):
        manifest = _build_agent_manifest([ALPHA])
        assert "Alpha agent for math" in manifest

    def test_includes_skills(self):
        manifest = _build_agent_manifest([ALPHA])
        assert "arithmetic" in manifest
        assert "algebra" in manifest

    def test_includes_endpoint(self):
        manifest = _build_agent_manifest([ALPHA])
        assert "http://example.com" in manifest

    def test_no_skills_shows_general_purpose(self):
        agent = make_agent("solo", skills=())
        manifest = _build_agent_manifest([agent])
        assert "general purpose" in manifest

    def test_empty_list(self):
        assert _build_agent_manifest([]) == ""


# ---------------------------------------------------------------------------
# _match_agent
# ---------------------------------------------------------------------------


class TestMatchAgent:
    def test_exact_match(self):
        assert _match_agent("alpha", [ALPHA, BETA]) is ALPHA

    def test_case_insensitive(self):
        assert _match_agent("BETA", [ALPHA, BETA]) is BETA

    def test_substring_in_chosen(self):
        # chosen_name contains agent name
        assert _match_agent("use the gamma agent please", [ALPHA, GAMMA]) is GAMMA

    def test_agent_name_contains_chosen(self):
        # agent name contains chosen fragment
        assert _match_agent("alph", [ALPHA, BETA]) is ALPHA

    def test_not_found_raises(self):
        with pytest.raises(AgentNotFoundError, match="ghost"):
            _match_agent("ghost", [ALPHA, BETA])

    def test_whitespace_stripped(self):
        assert _match_agent("  alpha  ", [ALPHA, BETA]) is ALPHA
