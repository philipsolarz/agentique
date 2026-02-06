"""Tests for extended router functionality (LLMRouter, WeightedKeywordRouter)."""

from __future__ import annotations

import pytest

from agentique.core.types import AgentInfo
from agentique.bridge.router import (
    AgentRouter,
    DirectRouter,
    KeywordRouter,
    LLMRouter,
    WeightedKeywordRouter,
)


def _agents() -> list[AgentInfo]:
    return [
        AgentInfo(name="calc", base_url="http://a", skills=("math", "calculator")),
        AgentInfo(name="text", base_url="http://b", skills=("nlp", "text")),
        AgentInfo(name="data", base_url="http://c", skills=("data", "analysis")),
    ]


# ---- WeightedKeywordRouter ----


def test_weighted_router_uses_weights():
    router = WeightedKeywordRouter(weights={"math": 5.0, "nlp": 1.0})
    agents = _agents()
    # "math" has weight 5.0, should win even with "text" in message
    result = router.select("text and math problem", agents)
    assert result.name == "calc"


def test_weighted_router_description_weight():
    router = WeightedKeywordRouter(description_weight=2.0)
    agents = [
        AgentInfo(name="a", base_url="http://a", description="handles numbers"),
        AgentInfo(name="b", base_url="http://b", description="handles text analysis"),
    ]
    result = router.select("I need text analysis", agents)
    assert result.name == "b"


def test_weighted_router_fallback_to_first():
    router = WeightedKeywordRouter()
    agents = _agents()
    result = router.select("completely unrelated query", agents)
    assert result.name == "calc"  # First agent as fallback


# ---- LLMRouter ----


def test_llm_router_sync_fallback():
    """Sync select() should use the keyword fallback."""
    router = LLMRouter()
    agents = _agents()
    result = router.select("do some math", agents)
    assert result.name == "calc"


@pytest.mark.asyncio
async def test_llm_router_no_ctx_fallback():
    """aselect() without ctx should fall back to keywords."""
    router = LLMRouter()
    agents = _agents()
    result = await router.aselect("do some math", agents, ctx=None)
    assert result.name == "calc"


@pytest.mark.asyncio
async def test_llm_router_with_mock_ctx():
    """aselect() with a mock ctx.sample() should use LLM routing."""

    class MockCtx:
        async def sample(self, prompt, system_prompt=None, **kwargs):
            return "text"  # Simulate LLM choosing "text" agent

    router = LLMRouter()
    agents = _agents()
    result = await router.aselect("anything", agents, ctx=MockCtx())
    assert result.name == "text"


@pytest.mark.asyncio
async def test_llm_router_fuzzy_match():
    """LLM returning extra text should still match via fuzzy matching."""

    class MockCtx:
        async def sample(self, prompt, system_prompt=None, **kwargs):
            return "I think data would be the best choice"

    router = LLMRouter()
    agents = _agents()
    result = await router.aselect("anything", agents, ctx=MockCtx())
    assert result.name == "data"


@pytest.mark.asyncio
async def test_llm_router_unknown_response_fallback():
    """LLM returning an unknown name should fall back to keywords."""

    class MockCtx:
        async def sample(self, prompt, system_prompt=None, **kwargs):
            return "nonexistent-agent"

    router = LLMRouter()
    agents = _agents()
    result = await router.aselect("do some math", agents, ctx=MockCtx())
    assert result.name == "calc"  # Keyword fallback


@pytest.mark.asyncio
async def test_llm_router_error_fallback():
    """If ctx.sample() raises, should fall back to keywords."""

    class FailingCtx:
        async def sample(self, prompt, system_prompt=None, **kwargs):
            raise RuntimeError("sampling failed")

    router = LLMRouter()
    agents = _agents()
    result = await router.aselect("do some math", agents, ctx=FailingCtx())
    assert result.name == "calc"


@pytest.mark.asyncio
async def test_llm_router_structured_sampling_result():
    class Result:
        result = {"agent_name": "data"}

    class MockCtx:
        async def sample(self, prompt, system_prompt=None, **kwargs):
            assert "tools" in kwargs
            assert "result_type" in kwargs
            return Result()

    router = LLMRouter()
    agents = _agents()
    result = await router.aselect("analyze dataset", agents, ctx=MockCtx())
    assert result.name == "data"


# ---- AgentRouter.aresolve ----


@pytest.mark.asyncio
async def test_aresolve_by_name():
    router = AgentRouter(_agents())
    result = await router.aresolve(name="text")
    assert result.name == "text"


@pytest.mark.asyncio
async def test_aresolve_by_skill():
    router = AgentRouter(_agents())
    result = await router.aresolve(skill="data")
    assert result.name == "data"


@pytest.mark.asyncio
async def test_aresolve_with_llm_strategy():
    class MockCtx:
        async def sample(self, prompt, system_prompt=None, **kwargs):
            return "data"

    router = AgentRouter(_agents(), strategy=LLMRouter())
    result = await router.aresolve(message="analyze this", ctx=MockCtx())
    assert result.name == "data"


# ---- AgentRouter.unregister ----


def test_unregister_existing():
    router = AgentRouter(_agents())
    assert router.unregister("calc")
    assert len(router.list_agents()) == 2
    names = {a.name for a in router.list_agents()}
    assert "calc" not in names


def test_unregister_nonexistent():
    router = AgentRouter(_agents())
    assert not router.unregister("nonexistent")
    assert len(router.list_agents()) == 3


def test_unregister_default_shifts():
    router = AgentRouter(_agents())
    # Default is first agent ("calc")
    assert router.resolve().name == "calc"
    router.unregister("calc")
    # Default should shift to next available
    assert router.resolve().name in {"text", "data"}
