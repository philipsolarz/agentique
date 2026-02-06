"""Tests for ToolMapper implementations."""

from __future__ import annotations

from agentique.core.tool_mapper import (
    DefaultToolMapper,
    FlatHierarchyToolMapper,
    PerSkillToolMapper,
)
from agentique.core.types import AgentInfo


def _agent(name="calc", skills=("math", "stats"), **kw) -> AgentInfo:
    return AgentInfo(name=name, base_url="http://x", skills=tuple(skills), **kw)


# ---- DefaultToolMapper ----


def test_default_mapper_one_tool_per_agent():
    mapper = DefaultToolMapper()
    tools = mapper.map_tools(_agent())
    assert len(tools) == 1
    assert tools[0]["name"] == "calc"
    assert "message" in tools[0]["inputSchema"]["properties"]


def test_default_mapper_includes_skills_in_desc():
    mapper = DefaultToolMapper()
    tools = mapper.map_tools(_agent(skills=("a", "b")))
    assert "a, b" in tools[0]["description"]


def test_default_mapper_resource():
    mapper = DefaultToolMapper()
    resources = mapper.map_resources(_agent(name="test"))
    assert len(resources) == 1
    assert resources[0]["uri"] == "a2a://agents/test"


# ---- PerSkillToolMapper ----


def test_per_skill_mapper():
    mapper = PerSkillToolMapper()
    tools = mapper.map_tools(_agent(skills=("math", "text")))
    assert len(tools) == 2
    names = {t["name"] for t in tools}
    assert names == {"calc_math", "calc_text"}


def test_per_skill_mapper_no_skills_fallback():
    mapper = PerSkillToolMapper()
    tools = mapper.map_tools(_agent(skills=()))
    assert len(tools) == 1
    assert tools[0]["name"] == "calc"


def test_per_skill_mapper_custom_separator():
    mapper = PerSkillToolMapper(separator=".")
    tools = mapper.map_tools(_agent(skills=("math",)))
    assert tools[0]["name"] == "calc.math"


# ---- FlatHierarchyToolMapper ----


def test_flat_hierarchy_with_sub_agents():
    mapper = FlatHierarchyToolMapper()
    info = _agent(
        name="root",
        metadata={
            "sub_agents": [
                {"name": "Calculator", "description": "Does math"},
                {"name": "TextProc", "description": "Text stuff"},
            ]
        },
    )
    tools = mapper.map_tools(info)
    assert len(tools) == 2
    names = {t["name"] for t in tools}
    assert names == {"root_Calculator", "root_TextProc"}


def test_flat_hierarchy_no_sub_agents_fallback():
    mapper = FlatHierarchyToolMapper()
    tools = mapper.map_tools(_agent())
    assert len(tools) == 1
    assert tools[0]["name"] == "calc"
