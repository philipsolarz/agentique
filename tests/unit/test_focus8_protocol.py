"""Tests for Focus #8 protocol compliance and cleanup items.

Covers:
  - 8e: A2A-Version header sent by A2AClientPool
  - 8a: requires_decomposition field removed from RoutingDecision
  - 8b: POLICY_CONTEXT_URI removed from extensions
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ---------------------------------------------------------------------------
# 8e: A2A-Version header
# ---------------------------------------------------------------------------


def test_client_pool_sends_a2a_version_header():
    """A2AClientPool adds A2A-Version header to all outgoing httpx requests."""
    import httpx
    from agentique.adapters.a2a.client import A2AClientPool

    pool = A2AClientPool(timeout=5.0)

    # Capture what httpx.AsyncClient is called with
    captured_kwargs: dict = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    with patch("httpx.AsyncClient", FakeAsyncClient):
        with patch("a2a.client.ClientFactory.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = MagicMock()
            import asyncio
            asyncio.run(pool.get("http://example.com"))

    assert "headers" in captured_kwargs, (
        f"Expected 'headers' in AsyncClient kwargs: {captured_kwargs}"
    )
    assert captured_kwargs["headers"].get("A2A-Version") == "0.3", (
        f"Expected A2A-Version: 0.3 in headers: {captured_kwargs['headers']}"
    )


def test_client_pool_default_a2a_version_is_0_3():
    """Default A2A protocol version is 0.3."""
    from agentique.adapters.a2a.client import A2AClientPool

    pool = A2AClientPool()
    assert pool._a2a_protocol_version == "0.3"


def test_client_pool_custom_a2a_version():
    """A2AClientPool accepts a custom a2a_protocol_version."""
    from agentique.adapters.a2a.client import A2AClientPool

    pool = A2AClientPool(a2a_protocol_version="0.4")
    assert pool._a2a_protocol_version == "0.4"


# ---------------------------------------------------------------------------
# 8a: requires_decomposition field removed
# ---------------------------------------------------------------------------


def test_routing_decision_has_no_requires_decomposition():
    """RoutingDecision no longer has a requires_decomposition field."""
    from agentique.bridge.router import RoutingDecision

    d = RoutingDecision(agent_id="alpha", confidence=0.9, reasoning="test")
    assert not hasattr(d, "requires_decomposition"), (
        "requires_decomposition was removed — it should not be on RoutingDecision"
    )


def test_routing_decision_fields_are_exactly():
    """RoutingDecision has exactly the expected fields (no extras)."""
    from agentique.bridge.router import RoutingDecision

    fields = set(RoutingDecision.model_fields.keys())
    expected = {"agent_id", "confidence", "reasoning", "fallback_agents"}
    assert fields == expected, (
        f"Unexpected RoutingDecision fields: {fields - expected}; "
        f"Missing fields: {expected - fields}"
    )


def test_routing_decision_ignores_requires_decomposition_if_passed():
    """Passing requires_decomposition to RoutingDecision raises or ignores it."""
    from agentique.bridge.router import RoutingDecision
    from pydantic import ValidationError

    # Pydantic v2 with model_config extra="ignore" will drop extra fields
    # or raise depending on config. Either way, the field should NOT appear on the model.
    try:
        d = RoutingDecision(
            agent_id="alpha",
            confidence=0.9,
            reasoning="test",
            requires_decomposition=True,  # Should be ignored or raise
        )
        # If it didn't raise, the field should not be present
        assert not hasattr(d, "requires_decomposition") or not d.model_fields_set.issuperset(
            {"requires_decomposition"}
        )
    except (ValidationError, TypeError):
        pass  # Acceptable — field rejected


# ---------------------------------------------------------------------------
# 8b: POLICY_CONTEXT_URI removed
# ---------------------------------------------------------------------------


def test_extensions_module_has_no_policy_context():
    """pack_policy_context and unpack_policy_context no longer exist."""
    import agentique.extensions as ext

    assert not hasattr(ext, "pack_policy_context"), (
        "pack_policy_context was removed — should not exist"
    )
    assert not hasattr(ext, "unpack_policy_context"), (
        "unpack_policy_context was removed — should not exist"
    )
    assert not hasattr(ext, "POLICY_CONTEXT_URI"), (
        "POLICY_CONTEXT_URI was removed — should not exist"
    )


def test_all_extension_uris_has_three_entries():
    """ALL_EXTENSION_URIS has exactly 3 entries after removing policy-context."""
    from agentique.extensions import ALL_EXTENSION_URIS

    assert len(ALL_EXTENSION_URIS) == 3, (
        f"Expected 3 extension URIs, got {len(ALL_EXTENSION_URIS)}: {ALL_EXTENSION_URIS}"
    )


def test_capabilities_resource_omits_policy_context():
    """a2a://capabilities resource does not advertise POLICY_CONTEXT_URI."""
    import asyncio
    import json
    from agentique.core.types import AgentInfo
    from agentique.server import create_server

    agent = AgentInfo(name="test", base_url="http://test.example.com")
    server = create_server(agents=[agent])

    async def _read():
        return await server.read_resource("a2a://capabilities")

    raw = asyncio.run(_read())
    content = raw.contents[0].content
    data = json.loads(content)

    extensions = data.get("extensions", [])
    assert "com.agentique/policy-context" not in extensions
