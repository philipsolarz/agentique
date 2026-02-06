"""Default ToolMapper implementations.

Provides two built-in mappers:

- ``DefaultToolMapper`` — one MCP tool per agent (current behaviour)
- ``PerSkillToolMapper`` — one MCP tool per agent skill

Users can implement the ``ToolMapper`` protocol with custom logic
(e.g. flattening sub-agent hierarchies, custom naming conventions).
"""

from __future__ import annotations

from typing import Any

from .types import AgentInfo


class DefaultToolMapper:
    """Creates one MCP tool per agent.

    This is the default behaviour: each registered agent becomes a single
    MCP tool that accepts a ``message`` parameter.
    """

    def map_tools(self, agent: AgentInfo) -> list[dict[str, Any]]:
        desc = agent.description or f"Send a message to the '{agent.name}' agent."
        skills_note = ""
        if agent.skills:
            skills_note = f" Skills: {', '.join(agent.skills)}."
        return [
            {
                "name": agent.name,
                "description": f"{desc}{skills_note}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send to the agent",
                        },
                    },
                    "required": ["message"],
                },
            }
        ]

    def map_resources(self, agent: AgentInfo) -> list[dict[str, Any]]:
        return [
            {
                "uri": f"a2a://agents/{agent.name}",
                "name": f"Agent: {agent.name}",
                "description": agent.description or f"Details for agent '{agent.name}'",
                "mimeType": "application/json",
            }
        ]

    def map_prompts(self, agent: AgentInfo) -> list[dict[str, Any]]:
        return []


class PerSkillToolMapper:
    """Creates one MCP tool per agent skill.

    If an agent declares skills ``["math", "text"]``, this mapper
    generates two tools: ``{agent}_math`` and ``{agent}_text``.

    When the agent declares no skills, it falls back to a single
    tool named after the agent (same as ``DefaultToolMapper``).
    """

    def __init__(self, *, separator: str = "_") -> None:
        self._sep = separator

    def map_tools(self, agent: AgentInfo) -> list[dict[str, Any]]:
        if not agent.skills:
            return DefaultToolMapper().map_tools(agent)

        tools: list[dict[str, Any]] = []
        for skill in agent.skills:
            tool_name = f"{agent.name}{self._sep}{skill}"
            tools.append(
                {
                    "name": tool_name,
                    "description": (
                        f"Use the '{skill}' capability of agent '{agent.name}'."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": (
                                    f"Message for the '{skill}' skill of '{agent.name}'"
                                ),
                            },
                        },
                        "required": ["message"],
                    },
                }
            )
        return tools

    def map_resources(self, agent: AgentInfo) -> list[dict[str, Any]]:
        return DefaultToolMapper().map_resources(agent)

    def map_prompts(self, agent: AgentInfo) -> list[dict[str, Any]]:
        return []


class FlatHierarchyToolMapper:
    """Creates tools from the sub-agent metadata in agent cards.

    If the agent card contains ``metadata.sub_agents``, each sub-agent
    is exposed as an individual MCP tool. Otherwise falls back to
    ``DefaultToolMapper`` behaviour.
    """

    def __init__(self, *, separator: str = "_") -> None:
        self._sep = separator

    def map_tools(self, agent: AgentInfo) -> list[dict[str, Any]]:
        sub_agents = agent.metadata.get("sub_agents", [])
        if not sub_agents:
            return DefaultToolMapper().map_tools(agent)

        tools: list[dict[str, Any]] = []
        for sub in sub_agents:
            if not isinstance(sub, dict):
                continue
            sub_name = sub.get("name", "unknown")
            sub_desc = sub.get("description", "")
            tool_name = f"{agent.name}{self._sep}{sub_name}"
            tools.append(
                {
                    "name": tool_name,
                    "description": sub_desc or f"Sub-agent '{sub_name}' of '{agent.name}'",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": f"Message for sub-agent '{sub_name}'",
                            },
                        },
                        "required": ["message"],
                    },
                }
            )
        return tools

    def map_resources(self, agent: AgentInfo) -> list[dict[str, Any]]:
        return DefaultToolMapper().map_resources(agent)

    def map_prompts(self, agent: AgentInfo) -> list[dict[str, Any]]:
        return []
