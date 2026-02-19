"""Tests for the ``a2a://agent/{name}`` per-agent resource."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentique.core.types import AgentInfo
from agentique.server import create_server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server(agents=None):
    if agents is None:
        agents = [
            AgentInfo(name="alpha", base_url="http://alpha:9000", description="Alpha agent"),
            AgentInfo(name="beta", base_url="http://beta:9000"),
        ]
    with patch("agentique.server.A2AAgentAdapter") as mock_adapter_cls:
        mock_adapter = MagicMock()
        mock_adapter.get_agent_card = AsyncMock(return_value=None)
        mock_adapter.close = AsyncMock()
        mock_adapter_cls.return_value = mock_adapter
        return create_server(agents=agents)


def _read_resource(mcp, uri: str) -> str:
    """Read an MCP resource by URI synchronously."""
    result = asyncio.run(mcp.read_resource(uri))
    return result.contents[0].content


# ---------------------------------------------------------------------------
# Resource registration
# ---------------------------------------------------------------------------


def test_agent_card_resource_uri_template_registered():
    """``a2a://agent/{name}`` resource template must be discoverable."""
    mcp = _make_server()
    templates = asyncio.run(mcp.list_resource_templates())
    uris = [t.uri_template for t in templates]
    assert any("a2a://agent/" in u for u in uris), (
        f"Expected 'a2a://agent/{{name}}' resource template; got: {uris}"
    )


# ---------------------------------------------------------------------------
# Resource content — known agent
# ---------------------------------------------------------------------------


def test_agent_card_resource_returns_valid_json():
    mcp = _make_server()
    raw = _read_resource(mcp, "a2a://agent/alpha")
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)


def test_agent_card_resource_has_name_and_base_url():
    mcp = _make_server()
    data = json.loads(_read_resource(mcp, "a2a://agent/alpha"))

    assert data["name"] == "alpha"
    assert data["base_url"] == "http://alpha:9000"


def test_agent_card_resource_has_description():
    mcp = _make_server()
    data = json.loads(_read_resource(mcp, "a2a://agent/alpha"))

    assert data.get("description") == "Alpha agent"


def test_agent_card_resource_has_skills_list():
    mcp = _make_server()
    data = json.loads(_read_resource(mcp, "a2a://agent/alpha"))

    assert "skills" in data
    assert isinstance(data["skills"], list)


def test_agent_card_resource_has_security_schemes_key():
    """``security_schemes`` key must always be present (empty dict when none declared)."""
    mcp = _make_server()
    data = json.loads(_read_resource(mcp, "a2a://agent/alpha"))

    assert "security_schemes" in data
    assert isinstance(data["security_schemes"], dict)


# ---------------------------------------------------------------------------
# Resource content — unknown agent
# ---------------------------------------------------------------------------


def test_agent_card_resource_unknown_agent_returns_error():
    mcp = _make_server()
    raw = _read_resource(mcp, "a2a://agent/does-not-exist")
    data = json.loads(raw)

    assert "error" in data
    assert "does-not-exist" in data["error"]


# ---------------------------------------------------------------------------
# Security schemes surfaced via resource
# ---------------------------------------------------------------------------


def test_agent_card_resource_includes_security_scheme_when_cached():
    """When the adapter returns a card with security_schemes, they appear in the resource."""
    from unittest.mock import AsyncMock, MagicMock, patch

    agents = [AgentInfo(name="secure", base_url="http://secure:9000")]
    card_dict = {
        "name": "secure",
        "url": "http://secure:9000",
        "version": "1.0",
        "skills": [],
        "security_schemes": {
            "apiKey": {"type": "apiKey", "name": "X-Key", "in": "header"}
        },
    }
    with patch("agentique.server.A2AAgentAdapter") as mock_cls:
        mock_adapter = MagicMock()
        mock_adapter.get_agent_card = AsyncMock(return_value=card_dict)
        mock_adapter.close = AsyncMock()
        mock_cls.return_value = mock_adapter
        mcp = create_server(agents=agents)

    data = json.loads(_read_resource(mcp, "a2a://agent/secure"))
    assert "apiKey" in data["security_schemes"]
    assert data["security_schemes"]["apiKey"]["type"] == "api_key"
