"""Tests for the enhanced LLMRouter with structured sampling."""

from __future__ import annotations

from typing import Any

import pytest

from agentique.bridge.router import AgentRouter, LLMRouter, KeywordRouter
from agentique.core.types import AgentInfo


# ---- Fixtures ----


@pytest.fixture
def agents() -> list[AgentInfo]:
    return [
        AgentInfo(
            name="calculator",
            base_url="http://localhost:9001",
            skills=("math", "calculator"),
            description="A calculator agent",
        ),
        AgentInfo(
            name="writer",
            base_url="http://localhost:9002",
            skills=("text", "writing"),
            description="A writing agent",
        ),
    ]


# ---- Mock sampling contexts ----


class MockStructuredSampleResult:
    """Mock result from ctx.sample() with result_type."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.text = result


class MockSampleResult:
    """Mock result from ctx.sample() without result_type (plain text)."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


class MockSamplingContext:
    """Mock FastMCP Context with sample() capability."""

    def __init__(
        self,
        response: str = "calculator",
        *,
        supports_structured: bool = True,
    ) -> None:
        self._response = response
        self._supports_structured = supports_structured
        self.sample_calls: list[dict[str, Any]] = []

    async def sample(
        self,
        messages: Any,
        *,
        system_prompt: str | None = None,
        result_type: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.sample_calls.append({
            "messages": messages,
            "system_prompt": system_prompt,
            "result_type": result_type,
        })

        if result_type is not None and self._supports_structured:
            return MockStructuredSampleResult(self._response)
        elif result_type is not None and not self._supports_structured:
            raise TypeError("result_type not supported")
        else:
            return MockSampleResult(self._response)


class MockFailingContext:
    """Mock context where sample() always fails."""

    async def sample(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Sampling failed")


# ---- Tests ----


@pytest.mark.asyncio
async def test_llm_router_structured_sampling(agents):
    """LLMRouter should use structured sampling with result_type."""
    ctx = MockSamplingContext(response="calculator")
    router = LLMRouter()

    result = await router.aselect("what is 2+2?", agents, ctx=ctx)

    assert result.name == "calculator"
    # Should have called sample with result_type
    assert len(ctx.sample_calls) == 1
    assert ctx.sample_calls[0]["result_type"] is not None


@pytest.mark.asyncio
async def test_llm_router_fallback_to_text_sampling(agents):
    """LLMRouter should fall back to text sampling when structured fails."""
    ctx = MockSamplingContext(
        response="writer",
        supports_structured=False,
    )
    router = LLMRouter()

    result = await router.aselect("write a poem", agents, ctx=ctx)

    assert result.name == "writer"
    # Should have been called twice: first structured (fails), then text
    assert len(ctx.sample_calls) == 2


@pytest.mark.asyncio
async def test_llm_router_complete_failure_falls_back(agents):
    """LLMRouter should fall back to keyword routing on total failure."""
    ctx = MockFailingContext()
    router = LLMRouter()

    result = await router.aselect("calculate 2+2", agents, ctx=ctx)

    # Should fall back to keyword router which matches "calculator"
    assert result.name == "calculator"


@pytest.mark.asyncio
async def test_llm_router_no_ctx_uses_keywords(agents):
    """LLMRouter without ctx should use keyword fallback."""
    router = LLMRouter()

    result = await router.aselect("math problem", agents, ctx=None)

    # KeywordRouter fallback: "math" matches calculator's skills
    assert result.name == "calculator"


@pytest.mark.asyncio
async def test_aresolve_uses_structured_sampling(agents):
    """AgentRouter.aresolve should use LLMRouter's structured sampling."""
    ctx = MockSamplingContext(response="writer")
    router = AgentRouter(agents, strategy=LLMRouter())

    result = await router.aresolve(message="write a story", ctx=ctx)

    assert result.name == "writer"
    assert len(ctx.sample_calls) >= 1
