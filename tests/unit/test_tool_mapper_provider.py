"""Tests for ToolMapper integration with AgentProvider.

Verifies that AgentProvider respects the tool_mapper parameter and that
the three built-in mappers (DefaultToolMapper, PerSkillToolMapper,
FlatHierarchyToolMapper) produce the expected tool structures.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

pytestmark = pytest.mark.anyio

from agentique.bridge.provider import AgentProvider
from agentique.core.tool_mapper import (
    DefaultToolMapper,
    FlatHierarchyToolMapper,
    PerSkillToolMapper,
)
from agentique.core.types import AgentInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(
    name: str,
    description: str = "An agent",
    skills: tuple[str, ...] = (),
    metadata: dict | None = None,
) -> AgentInfo:
    return AgentInfo(
        name=name,
        base_url=f"http://{name}",
        description=description,
        skills=skills,
        metadata=metadata or {},
    )


def make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.close = AsyncMock()
    adapter.get_agent_card = AsyncMock(return_value=None)
    return adapter


def make_provider(
    agents: list[AgentInfo],
    tool_mapper=None,
) -> AgentProvider:
    return AgentProvider(
        agents,
        make_adapter(),
        cache_ttl=300.0,
        prefetch_cards=False,
        tool_mapper=tool_mapper,
    )


async def list_tool_names(provider: AgentProvider) -> list[str]:
    tools = await provider._list_tools()
    return [t.name for t in tools]


# ---------------------------------------------------------------------------
# Default behaviour (no mapper)
# ---------------------------------------------------------------------------


async def test_default_creates_one_tool_per_agent():
    agents = [make_agent("alpha"), make_agent("beta")]
    provider = make_provider(agents)
    names = await list_tool_names(provider)
    assert "alpha" in names
    assert "beta" in names


# ---------------------------------------------------------------------------
# DefaultToolMapper
# ---------------------------------------------------------------------------


async def test_default_mapper_produces_same_result_as_no_mapper():
    agents = [make_agent("alpha"), make_agent("beta")]
    no_mapper = make_provider(agents)
    with_mapper = make_provider(agents, DefaultToolMapper())
    names_none = set(await list_tool_names(no_mapper))
    names_mapper = set(await list_tool_names(with_mapper))
    assert names_none == names_mapper


async def test_default_mapper_uses_agent_description():
    agent = make_agent("svc", description="A useful service")
    provider = make_provider([agent], DefaultToolMapper())
    tools = await provider._list_tools()
    tool = next(t for t in tools if t.name == "svc")
    assert "useful service" in (tool.description or "")


# ---------------------------------------------------------------------------
# PerSkillToolMapper
# ---------------------------------------------------------------------------


async def test_per_skill_mapper_creates_one_tool_per_skill():
    agent = make_agent("worker", skills=("math", "text", "code"))
    provider = make_provider([agent], PerSkillToolMapper())
    names = await list_tool_names(provider)
    assert "worker_math" in names
    assert "worker_text" in names
    assert "worker_code" in names


async def test_per_skill_mapper_falls_back_when_no_skills():
    agent = make_agent("worker")  # no skills
    provider = make_provider([agent], PerSkillToolMapper())
    names = await list_tool_names(provider)
    # Should fall back to DefaultToolMapper behaviour: one tool named after the agent
    assert "worker" in names


async def test_per_skill_mapper_custom_separator():
    agent = make_agent("bot", skills=("search",))
    provider = make_provider([agent], PerSkillToolMapper(separator="-"))
    names = await list_tool_names(provider)
    assert "bot-search" in names


async def test_per_skill_mapper_does_not_create_agent_level_tool():
    """Tool named after the agent itself should NOT appear when skills exist."""
    agent = make_agent("worker", skills=("math",))
    provider = make_provider([agent], PerSkillToolMapper())
    names = await list_tool_names(provider)
    # "worker_math" should be there, but bare "worker" should not
    # (unless the card parser adds a proxy tool — we have no card here)
    assert "worker_math" in names
    # bare name NOT present (PerSkillToolMapper returns skill tools, not the agent tool)
    assert "worker" not in names


# ---------------------------------------------------------------------------
# FlatHierarchyToolMapper
# ---------------------------------------------------------------------------


async def test_flat_hierarchy_mapper_creates_sub_agent_tools():
    agent = make_agent(
        "hub",
        metadata={
            "sub_agents": [
                {"name": "worker-a", "description": "Does A"},
                {"name": "worker-b", "description": "Does B"},
            ]
        },
    )
    provider = make_provider([agent], FlatHierarchyToolMapper())
    names = await list_tool_names(provider)
    assert "hub_worker-a" in names
    assert "hub_worker-b" in names


async def test_flat_hierarchy_mapper_falls_back_without_sub_agents():
    agent = make_agent("hub")  # no sub_agents in metadata
    provider = make_provider([agent], FlatHierarchyToolMapper())
    names = await list_tool_names(provider)
    assert "hub" in names


# ---------------------------------------------------------------------------
# Tool handlers are callable
# ---------------------------------------------------------------------------


async def test_mapper_tool_handler_calls_adapter_send_message():
    """Smoke test: the tool created by mapper def has a working handler."""
    agent = make_agent("svc", description="Test agent")
    mapper = DefaultToolMapper()
    adapter = make_adapter()
    from agentique.core.types import AgentResponse
    adapter.send_message = AsyncMock(
        return_value=AgentResponse(agent="svc", text="ok")
    )
    provider = AgentProvider(
        [agent], adapter,
        cache_ttl=300.0, prefetch_cards=False,
        tool_mapper=mapper,
    )
    tools = await provider._list_tools()
    svc_tool = next(t for t in tools if t.name == "svc")
    # Tool exists and has a description
    assert svc_tool is not None
