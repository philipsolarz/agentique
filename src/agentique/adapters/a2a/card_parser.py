"""Parse A2A Agent Cards into agentique core types.

Handles extraction of MCP tool/resource/prompt definitions from the
extensions mechanism in agent cards, plus sub-agent hierarchy metadata.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentique.core.types import AgentHierarchy, AgentInfo

logger = logging.getLogger(__name__)


class A2ACardParser:
    """Parse A2A agent cards into structured component definitions."""

    def extract_components(
        self, card: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (tools, prompts, resources) from an agent card."""
        payload = self._to_dict(card)

        tools = self._collect(payload, "mcp_tools", "mcpTools")
        prompts = self._collect(payload, "mcp_prompts", "mcpPrompts")
        resources = self._collect(payload, "mcp_resources", "mcpResources")

        # Also check extensions
        for ext in self._extensions(card):
            params = self._ext_params(ext)
            if isinstance(params, dict):
                tools.extend(self._collect(params, "mcp_tools", "mcpTools"))
                prompts.extend(self._collect(params, "mcp_prompts", "mcpPrompts"))
                resources.extend(self._collect(params, "mcp_resources", "mcpResources"))

        return tools, prompts, resources

    def build_hierarchy(self, root_name: str, card: Any) -> AgentHierarchy:
        """Build a sub-agent hierarchy from card metadata."""
        hierarchy = AgentHierarchy(root=root_name)
        payload = self._to_dict(card)

        # Skills as child agents
        skills = payload.get("skills") or []
        for skill in skills:
            if isinstance(skill, dict):
                name = skill.get("name") or skill.get("id")
                desc = skill.get("description")
            else:
                name = getattr(skill, "name", None) or getattr(skill, "id", None)
                desc = getattr(skill, "description", None)
            if name:
                hierarchy.add_agent(name, description=desc, parent=root_name)

        # Explicit sub_agents in metadata
        metadata = payload.get("metadata") or {}
        for sub in metadata.get("sub_agents", []):
            if isinstance(sub, dict):
                hierarchy.add_agent(
                    sub.get("name", "unknown"),
                    description=sub.get("description"),
                    skills=sub.get("skills", []),
                    parent=root_name,
                )

        return hierarchy

    def card_to_agent_info(
        self, card: Any, *, fallback_name: str = "unknown", fallback_url: str = ""
    ) -> AgentInfo:
        """Convert an agent card to an ``AgentInfo``."""
        payload = self._to_dict(card)
        name = payload.get("name") or fallback_name
        url = payload.get("url") or fallback_url
        description = payload.get("description")

        skills_raw = payload.get("skills") or []
        skill_names: list[str] = []
        for s in skills_raw:
            if isinstance(s, dict):
                skill_names.append(s.get("name") or s.get("id", ""))
            elif isinstance(s, str):
                skill_names.append(s)
            else:
                sn = getattr(s, "name", None) or getattr(s, "id", None)
                if sn:
                    skill_names.append(sn)

        return AgentInfo(
            name=name,
            base_url=url,
            description=description,
            skills=tuple(skill_names),
            metadata=payload.get("metadata") or {},
        )

    # ---- internal helpers ----

    def _to_dict(self, card: Any) -> dict[str, Any]:
        if card is None:
            return {}
        if isinstance(card, dict):
            return dict(card)
        result: dict[str, Any] = {}
        extra = getattr(card, "model_extra", None) or getattr(card, "__pydantic_extra__", None)
        if isinstance(extra, dict):
            result.update(extra)
        if hasattr(card, "model_dump"):
            try:
                result.update(card.model_dump())
            except Exception:
                pass
        elif hasattr(card, "dict"):
            try:
                result.update(card.dict())
            except Exception:
                pass
        return result

    def _extensions(self, card: Any) -> list[Any]:
        payload = self._to_dict(card)
        caps = payload.get("capabilities") or {}
        if isinstance(caps, dict):
            exts = caps.get("extensions")
        else:
            exts = getattr(caps, "extensions", None)
        return list(exts) if exts else []

    def _ext_params(self, ext: Any) -> Any:
        for name in ("params", "parameters"):
            val = ext.get(name) if isinstance(ext, dict) else getattr(ext, name, None)
            if val is not None:
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except Exception:
                        return None
                return val
        return None

    def _collect(self, d: dict[str, Any], *keys: str) -> list[Any]:
        for key in keys:
            val = d.get(key)
            if val is not None:
                return list(val) if isinstance(val, (list, tuple)) else [val]
        return []
