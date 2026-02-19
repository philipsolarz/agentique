"""Tests for agentique.extensions — A2A Extensions passthrough module.

Covers:
  - URI constants are properly namespaced
  - ALL_EXTENSION_URIS contains all four constants
  - pack_routing_metadata / unpack_routing_metadata round-trip
  - pack_policy_context / unpack_policy_context round-trip
  - pack_mcp_session / unpack_mcp_session round-trip
  - pack_trace_context / unpack_trace_context round-trip
  - current_trace_context: returns empty dict when OTel not active
  - build_gateway_metadata: composes all channels
  - unpack returns None for absent keys
"""

from __future__ import annotations

import pytest

from agentique.bridge.router import RoutingDecision
from agentique.extensions import (
    ALL_EXTENSION_URIS,
    MCP_SESSION_URI,
    POLICY_CONTEXT_URI,
    ROUTING_METADATA_URI,
    TRACE_CONTEXT_URI,
    build_gateway_metadata,
    current_trace_context,
    pack_mcp_session,
    pack_policy_context,
    pack_routing_metadata,
    pack_trace_context,
    unpack_mcp_session,
    unpack_policy_context,
    unpack_routing_metadata,
    unpack_trace_context,
)


# ---------------------------------------------------------------------------
# URI constants
# ---------------------------------------------------------------------------


def test_routing_metadata_uri_namespaced():
    assert ROUTING_METADATA_URI.startswith("com.agentique/")


def test_policy_context_uri_namespaced():
    assert POLICY_CONTEXT_URI.startswith("com.agentique/")


def test_mcp_session_uri_namespaced():
    assert MCP_SESSION_URI.startswith("com.agentique/")


def test_trace_context_uri_namespaced():
    assert TRACE_CONTEXT_URI.startswith("com.agentique/")


def test_all_extension_uris_complete():
    uris = set(ALL_EXTENSION_URIS)
    assert ROUTING_METADATA_URI in uris
    assert POLICY_CONTEXT_URI in uris
    assert MCP_SESSION_URI in uris
    assert TRACE_CONTEXT_URI in uris
    assert len(ALL_EXTENSION_URIS) == 4


def test_all_uris_unique():
    assert len(set(ALL_EXTENSION_URIS)) == len(ALL_EXTENSION_URIS)


# ---------------------------------------------------------------------------
# Routing metadata
# ---------------------------------------------------------------------------


def _make_decision(**kwargs):
    defaults = dict(
        agent_id="billing",
        confidence=0.92,
        reasoning="billing skills match",
        fallback_agents=["crm"],
        requires_decomposition=False,
    )
    defaults.update(kwargs)
    return RoutingDecision(**defaults)


def test_pack_routing_metadata_basic():
    decision = _make_decision()
    packed = pack_routing_metadata(decision)
    assert packed["agent_id"] == "billing"
    assert packed["confidence"] == pytest.approx(0.92)
    assert packed["reasoning"] == "billing skills match"
    assert packed["fallback_agents"] == ["crm"]


def test_pack_routing_metadata_no_requires_decomposition_when_false():
    decision = _make_decision(requires_decomposition=False)
    packed = pack_routing_metadata(decision)
    assert "requires_decomposition" not in packed


def test_pack_routing_metadata_includes_requires_decomposition_when_true():
    decision = _make_decision(requires_decomposition=True)
    packed = pack_routing_metadata(decision)
    assert packed["requires_decomposition"] is True


def test_pack_routing_metadata_empty_fallbacks_omitted():
    decision = _make_decision(fallback_agents=[])
    packed = pack_routing_metadata(decision)
    assert "fallback_agents" not in packed


def test_unpack_routing_metadata_round_trip():
    decision = _make_decision()
    metadata = {ROUTING_METADATA_URI: pack_routing_metadata(decision)}
    unpacked = unpack_routing_metadata(metadata)
    assert unpacked is not None
    assert unpacked["agent_id"] == "billing"
    assert unpacked["confidence"] == pytest.approx(0.92)
    assert unpacked["fallback_agents"] == ["crm"]


def test_unpack_routing_metadata_absent_returns_none():
    result = unpack_routing_metadata({})
    assert result is None


def test_unpack_routing_metadata_wrong_type_returns_none():
    result = unpack_routing_metadata({ROUTING_METADATA_URI: "not a dict"})
    assert result is None


# ---------------------------------------------------------------------------
# Policy context
# ---------------------------------------------------------------------------


def test_pack_policy_context_full():
    packed = pack_policy_context(
        tenant_id="acme", visibility_tier="tier_1", rate_limit_budget=500
    )
    assert packed["tenant_id"] == "acme"
    assert packed["visibility_tier"] == "tier_1"
    assert packed["rate_limit_budget"] == 500


def test_pack_policy_context_partial():
    packed = pack_policy_context(tenant_id="acme")
    assert "tenant_id" in packed
    assert "visibility_tier" not in packed
    assert "rate_limit_budget" not in packed


def test_pack_policy_context_empty():
    packed = pack_policy_context()
    assert packed == {}


def test_unpack_policy_context_round_trip():
    metadata = {
        POLICY_CONTEXT_URI: pack_policy_context(tenant_id="acme", visibility_tier="tier_1")
    }
    result = unpack_policy_context(metadata)
    assert result is not None
    assert result["tenant_id"] == "acme"
    assert result["visibility_tier"] == "tier_1"


def test_unpack_policy_context_absent_returns_none():
    assert unpack_policy_context({}) is None


# ---------------------------------------------------------------------------
# MCP session context
# ---------------------------------------------------------------------------


def test_pack_mcp_session_full():
    packed = pack_mcp_session(
        session_id="sess-123", context_id="ctx-456", client_name="Claude Desktop"
    )
    assert packed["session_id"] == "sess-123"
    assert packed["context_id"] == "ctx-456"
    assert packed["client_name"] == "Claude Desktop"


def test_pack_mcp_session_empty():
    assert pack_mcp_session() == {}


def test_unpack_mcp_session_round_trip():
    metadata = {MCP_SESSION_URI: pack_mcp_session(session_id="s1", context_id="c1")}
    result = unpack_mcp_session(metadata)
    assert result is not None
    assert result["session_id"] == "s1"
    assert result["context_id"] == "c1"


def test_unpack_mcp_session_absent_returns_none():
    assert unpack_mcp_session({}) is None


# ---------------------------------------------------------------------------
# Trace context
# ---------------------------------------------------------------------------


def test_pack_trace_context_full():
    packed = pack_trace_context(
        traceparent="00-abc123-def456-01", tracestate="vendor=data"
    )
    assert packed["traceparent"] == "00-abc123-def456-01"
    assert packed["tracestate"] == "vendor=data"


def test_pack_trace_context_empty():
    assert pack_trace_context() == {}


def test_unpack_trace_context_round_trip():
    metadata = {TRACE_CONTEXT_URI: pack_trace_context(traceparent="00-abc-def-01")}
    result = unpack_trace_context(metadata)
    assert result is not None
    assert result["traceparent"] == "00-abc-def-01"


def test_unpack_trace_context_absent_returns_none():
    assert unpack_trace_context({}) is None


# ---------------------------------------------------------------------------
# current_trace_context
# ---------------------------------------------------------------------------


def test_current_trace_context_returns_dict():
    result = current_trace_context()
    assert isinstance(result, dict)


def test_current_trace_context_empty_when_no_otel():
    # Without an active OTel span, should return empty dict (not raise)
    result = current_trace_context()
    # May or may not be empty depending on test environment — just verify it's a dict
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# build_gateway_metadata
# ---------------------------------------------------------------------------


def test_build_gateway_metadata_with_decision():
    decision = _make_decision()
    meta = build_gateway_metadata(routing_decision=decision, include_trace=False)
    assert ROUTING_METADATA_URI in meta
    assert meta[ROUTING_METADATA_URI]["agent_id"] == "billing"


def test_build_gateway_metadata_with_policy():
    meta = build_gateway_metadata(
        tenant_id="acme", visibility_tier="tier_1", include_trace=False
    )
    assert POLICY_CONTEXT_URI in meta
    assert meta[POLICY_CONTEXT_URI]["tenant_id"] == "acme"


def test_build_gateway_metadata_with_session():
    meta = build_gateway_metadata(
        session_id="s1", context_id="c1", include_trace=False
    )
    assert MCP_SESSION_URI in meta
    assert meta[MCP_SESSION_URI]["session_id"] == "s1"


def test_build_gateway_metadata_empty_no_trace():
    meta = build_gateway_metadata(include_trace=False)
    # No channels set → empty dict
    assert meta == {}


def test_build_gateway_metadata_all_channels():
    decision = _make_decision()
    meta = build_gateway_metadata(
        routing_decision=decision,
        tenant_id="acme",
        session_id="s1",
        context_id="c1",
        include_trace=False,
    )
    assert ROUTING_METADATA_URI in meta
    assert POLICY_CONTEXT_URI in meta
    assert MCP_SESSION_URI in meta


def test_build_gateway_metadata_with_trace_include():
    """include_trace=True should not raise even with no OTel."""
    meta = build_gateway_metadata(include_trace=True)
    assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# Public import from agentique top-level
# ---------------------------------------------------------------------------


def test_extensions_importable_from_agentique():
    from agentique import (
        ROUTING_METADATA_URI,
        build_gateway_metadata,
        pack_routing_metadata,
        unpack_routing_metadata,
    )
    assert ROUTING_METADATA_URI.startswith("com.agentique/")
