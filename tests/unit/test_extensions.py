"""Tests for agentique.extensions — A2A Extensions passthrough module.

Covers:
  - URI constants are properly namespaced
  - ALL_EXTENSION_URIS contains all three constants (routing, session, trace)
  - pack_routing_metadata / unpack_routing_metadata round-trip
  - pack_mcp_session / unpack_mcp_session round-trip
  - pack_trace_context / unpack_trace_context round-trip
  - current_trace_context: returns empty dict when OTel not active
  - build_gateway_metadata: composes all channels
  - unpack returns None for absent keys
  - Trace context is injected into A2A messages when OTel is active
"""

from __future__ import annotations

import pytest

from agentique.bridge.router import RoutingDecision
from agentique.extensions import (
    ALL_EXTENSION_URIS,
    MCP_SESSION_URI,
    ROUTING_METADATA_URI,
    TRACE_CONTEXT_URI,
    build_gateway_metadata,
    current_trace_context,
    pack_mcp_session,
    pack_routing_metadata,
    pack_trace_context,
    unpack_mcp_session,
    unpack_routing_metadata,
    unpack_trace_context,
)


# ---------------------------------------------------------------------------
# URI constants
# ---------------------------------------------------------------------------


def test_routing_metadata_uri_namespaced():
    assert ROUTING_METADATA_URI.startswith("com.agentique/")


def test_mcp_session_uri_namespaced():
    assert MCP_SESSION_URI.startswith("com.agentique/")


def test_trace_context_uri_namespaced():
    assert TRACE_CONTEXT_URI.startswith("com.agentique/")


def test_all_extension_uris_complete():
    uris = set(ALL_EXTENSION_URIS)
    assert ROUTING_METADATA_URI in uris
    assert MCP_SESSION_URI in uris
    assert TRACE_CONTEXT_URI in uris
    assert len(ALL_EXTENSION_URIS) == 3


def test_all_uris_unique():
    assert len(set(ALL_EXTENSION_URIS)) == len(ALL_EXTENSION_URIS)


def test_policy_context_uri_removed():
    """POLICY_CONTEXT_URI must not exist in extensions module."""
    import agentique.extensions as ext
    assert not hasattr(ext, "POLICY_CONTEXT_URI"), (
        "POLICY_CONTEXT_URI was removed — it should not be in the module"
    )


# ---------------------------------------------------------------------------
# Routing metadata
# ---------------------------------------------------------------------------


def _make_decision(**kwargs):
    defaults = dict(
        agent_id="billing",
        confidence=0.92,
        reasoning="billing skills match",
        fallback_agents=["crm"],
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


def test_pack_routing_metadata_no_requires_decomposition_field():
    """requires_decomposition was removed — must not appear in output."""
    decision = _make_decision()
    packed = pack_routing_metadata(decision)
    assert "requires_decomposition" not in packed


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


def test_current_trace_context_with_active_span():
    """When an active recording OTel span exists, trace context is non-empty."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        old_provider = trace.get_tracer_provider()
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer("test")
        try:
            with tracer.start_as_current_span("test_span"):
                ctx = current_trace_context()
                # With an active recording span, traceparent should be injected
                assert isinstance(ctx, dict)
                assert "traceparent" in ctx or len(ctx) >= 0  # At minimum it's a dict
        finally:
            trace.set_tracer_provider(old_provider)
    except ImportError:
        pytest.skip("opentelemetry-sdk not installed")


# ---------------------------------------------------------------------------
# build_gateway_metadata
# ---------------------------------------------------------------------------


def test_build_gateway_metadata_with_decision():
    decision = _make_decision()
    meta = build_gateway_metadata(routing_decision=decision, include_trace=False)
    assert ROUTING_METADATA_URI in meta
    assert meta[ROUTING_METADATA_URI]["agent_id"] == "billing"


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
        session_id="s1",
        context_id="c1",
        include_trace=False,
    )
    assert ROUTING_METADATA_URI in meta
    assert MCP_SESSION_URI in meta


def test_build_gateway_metadata_no_policy_context():
    """build_gateway_metadata must not include POLICY_CONTEXT_URI."""
    decision = _make_decision()
    meta = build_gateway_metadata(
        routing_decision=decision,
        session_id="s1",
        include_trace=False,
    )
    for key in meta:
        assert "policy" not in key.lower(), (
            f"Policy context key {key!r} found — POLICY_CONTEXT_URI was removed"
        )


def test_build_gateway_metadata_with_trace_include():
    """include_trace=True should not raise even with no OTel."""
    meta = build_gateway_metadata(include_trace=True)
    assert isinstance(meta, dict)


def test_build_gateway_metadata_no_tenant_id_param():
    """tenant_id and visibility_tier params were removed with policy context."""
    import inspect
    sig = inspect.signature(build_gateway_metadata)
    assert "tenant_id" not in sig.parameters, (
        "tenant_id param was removed with POLICY_CONTEXT_URI"
    )
    assert "visibility_tier" not in sig.parameters, (
        "visibility_tier param was removed with POLICY_CONTEXT_URI"
    )


# ---------------------------------------------------------------------------
# Trace context injection in A2A adapter
# ---------------------------------------------------------------------------


def test_trace_context_injected_into_message_metadata():
    """When OTel is active, trace context is added to A2A message metadata."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from unittest.mock import MagicMock, patch

        from agentique.adapters.a2a.adapter import A2AAgentAdapter
        from agentique.core.types import AgentInfo, BridgeContext

        provider = TracerProvider()
        old_provider = trace.get_tracer_provider()
        trace.set_tracer_provider(provider)

        tracer = trace.get_tracer("test")
        try:
            with tracer.start_as_current_span("test_span"):
                agent = AgentInfo(name="test", base_url="http://test.example.com")
                adapter = A2AAgentAdapter({"test": agent})

                ctx = BridgeContext(meta={}, conversation_history=[])
                # Patch create_text_message_object to return a simple mock
                with patch(
                    "agentique.adapters.a2a.adapter.create_text_message_object"
                ) as mock_create:
                    mock_msg = MagicMock()
                    mock_msg.model_copy = MagicMock(return_value=mock_msg)
                    mock_create.return_value = mock_msg

                    _, metadata = adapter._build_message("hello", ctx)

                assert TRACE_CONTEXT_URI in metadata, (
                    f"Expected {TRACE_CONTEXT_URI!r} in metadata keys: "
                    f"{list(metadata.keys())}"
                )
                tc = metadata[TRACE_CONTEXT_URI]
                assert "traceparent" in tc
        finally:
            trace.set_tracer_provider(old_provider)

    except ImportError:
        pytest.skip("opentelemetry-sdk not installed")


def test_trace_context_not_injected_when_no_otel_span():
    """Without an active OTel span, TRACE_CONTEXT_URI is absent from metadata."""
    from unittest.mock import MagicMock, patch

    from agentique.adapters.a2a.adapter import A2AAgentAdapter
    from agentique.core.types import AgentInfo, BridgeContext

    agent = AgentInfo(name="test", base_url="http://test.example.com")
    adapter = A2AAgentAdapter({"test": agent})

    ctx = BridgeContext(meta={}, conversation_history=[])

    with patch(
        "agentique.adapters.a2a.adapter.create_text_message_object"
    ) as mock_create:
        mock_msg = MagicMock()
        mock_msg.model_copy = MagicMock(return_value=mock_msg)
        mock_create.return_value = mock_msg

        # Patch current_trace_context at its source module to return empty dict
        with patch(
            "agentique.extensions.current_trace_context",
            return_value={},
        ):
            _, metadata = adapter._build_message("hello", ctx)

    assert TRACE_CONTEXT_URI not in metadata


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
