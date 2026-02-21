"""Agentique gateway extension definitions for A2A and MCP metadata passthrough.

Defines URI-keyed extension constants and helper functions that pack
gateway-internal metadata (routing decisions, MCP session info, trace context)
into A2A message ``metadata`` dicts and MCP tool ``_meta`` fields.

Extension URIs follow the reversed-domain convention:
``com.agentique/<channel>``

This matches both A2A's extension mechanism (``AgentExtension`` in the Agent
Card + ``metadata`` keyed by URI in messages) and MCP's Extensions framework
(``capabilities.extensions`` negotiation during initialization).

Usage — attaching routing metadata to an A2A message::

    from agentique.extensions import (
        ROUTING_METADATA_URI,
        pack_routing_metadata,
    )

    decision = RoutingDecision(
        agent_id="billing",
        confidence=0.92,
        reasoning="billing skills match invoice request",
        fallback_agents=["crm"],
    )
    metadata = {ROUTING_METADATA_URI: pack_routing_metadata(decision)}
    # Pass metadata to adapter.send_message(..., metadata=metadata)

Usage — extracting routing metadata from an incoming message::

    from agentique.extensions import ROUTING_METADATA_URI, unpack_routing_metadata

    routing = unpack_routing_metadata(message.metadata or {})
    if routing:
        print(routing.agent_id, routing.confidence)

Design notes
------------
- Packing functions return plain ``dict[str, Any]`` so they can be
  serialised without any Pydantic dependency at the call site.
- Unpacking functions return typed dicts (not Pydantic models) to avoid
  hard coupling between this module and the bridge layer.
- ``None`` values are omitted from packed dicts to minimise wire size.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Extension URI constants
# ---------------------------------------------------------------------------

#: Carries routing decision metadata (agent_id, confidence, reasoning,
#: fallback_agents).
ROUTING_METADATA_URI: str = "com.agentique/routing-metadata"

#: Carries MCP session context (session_id, client_capabilities_hash).
MCP_SESSION_URI: str = "com.agentique/mcp-session"

#: Carries OpenTelemetry W3C trace context (traceparent, tracestate).
TRACE_CONTEXT_URI: str = "com.agentique/trace-context"

#: Complete list of all Agentique extension URIs for capability negotiation.
ALL_EXTENSION_URIS: list[str] = [
    ROUTING_METADATA_URI,
    MCP_SESSION_URI,
    TRACE_CONTEXT_URI,
]


# ---------------------------------------------------------------------------
# Routing metadata
# ---------------------------------------------------------------------------


def pack_routing_metadata(decision: Any) -> dict[str, Any]:
    """Pack a ``RoutingDecision`` into an A2A extension metadata dict.

    Args:
        decision: A ``RoutingDecision`` instance (or any object with
            ``agent_id``, ``confidence``, ``reasoning``,
            ``fallback_agents`` attributes).

    Returns:
        A plain dict suitable for use as A2A message metadata under
        ``ROUTING_METADATA_URI``.
    """
    result: dict[str, Any] = {
        "agent_id": str(getattr(decision, "agent_id", "") or ""),
        "confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
    }
    reasoning = getattr(decision, "reasoning", None)
    if reasoning:
        result["reasoning"] = str(reasoning)
    fallbacks = getattr(decision, "fallback_agents", None)
    if fallbacks:
        result["fallback_agents"] = list(fallbacks)
    return result


def unpack_routing_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Extract routing metadata from an A2A metadata dict.

    Args:
        metadata: The full A2A message ``metadata`` dict.

    Returns:
        A plain dict with routing fields, or ``None`` if the extension
        key is absent.
    """
    payload = metadata.get(ROUTING_METADATA_URI)
    if not isinstance(payload, dict):
        return None
    return {
        "agent_id": payload.get("agent_id", ""),
        "confidence": float(payload.get("confidence", 0.0)),
        "reasoning": payload.get("reasoning", ""),
        "fallback_agents": list(payload.get("fallback_agents") or []),
    }


# ---------------------------------------------------------------------------
# MCP session context
# ---------------------------------------------------------------------------


def pack_mcp_session(
    *,
    session_id: str | None = None,
    context_id: str | None = None,
    client_name: str | None = None,
) -> dict[str, Any]:
    """Pack MCP session context into an A2A extension metadata dict.

    Enables A2A agents to correlate gateway sessions with their own
    task tracking and audit trails.

    Args:
        session_id: The MCP session ID (``MCP-Session-Id`` header value).
        context_id: The A2A context ID bound to this session.
        client_name: Identifying name of the MCP client (from initialization).

    Returns:
        A plain dict suitable for ``MCP_SESSION_URI`` in A2A metadata.
    """
    result: dict[str, Any] = {}
    if session_id is not None:
        result["session_id"] = session_id
    if context_id is not None:
        result["context_id"] = context_id
    if client_name is not None:
        result["client_name"] = client_name
    return result


def unpack_mcp_session(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Extract MCP session context from an A2A metadata dict."""
    payload = metadata.get(MCP_SESSION_URI)
    if not isinstance(payload, dict):
        return None
    return {
        "session_id": payload.get("session_id"),
        "context_id": payload.get("context_id"),
        "client_name": payload.get("client_name"),
    }


# ---------------------------------------------------------------------------
# Trace context (W3C traceparent / tracestate)
# ---------------------------------------------------------------------------


def pack_trace_context(
    *,
    traceparent: str | None = None,
    tracestate: str | None = None,
) -> dict[str, Any]:
    """Pack OpenTelemetry W3C trace context into an A2A extension metadata dict.

    Enables end-to-end distributed tracing from MCP client through the gateway
    to A2A agents without requiring the agents to understand OTel directly.

    Args:
        traceparent: W3C ``traceparent`` header value
            (``00-<trace_id>-<parent_id>-<flags>``).
        tracestate: W3C ``tracestate`` header value (vendor-specific state).

    Returns:
        A plain dict suitable for ``TRACE_CONTEXT_URI`` in A2A metadata.
    """
    result: dict[str, Any] = {}
    if traceparent:
        result["traceparent"] = traceparent
    if tracestate:
        result["tracestate"] = tracestate
    return result


def current_trace_context() -> dict[str, Any]:
    """Extract the current OTel span's W3C trace context.

    Returns an empty dict when OpenTelemetry is not configured or the
    current span is non-recording (silent no-op).

    Returns:
        A dict with ``traceparent`` (and optionally ``tracestate``) if a
        recording span is active, otherwise an empty dict.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.propagate import inject

        carrier: dict[str, str] = {}
        inject(carrier)
        return {k: v for k, v in carrier.items() if v}
    except Exception:
        return {}


def unpack_trace_context(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Extract W3C trace context from an A2A metadata dict."""
    payload = metadata.get(TRACE_CONTEXT_URI)
    if not isinstance(payload, dict):
        return None
    return {
        "traceparent": payload.get("traceparent"),
        "tracestate": payload.get("tracestate"),
    }


# ---------------------------------------------------------------------------
# Convenience: build a complete gateway metadata dict
# ---------------------------------------------------------------------------


def build_gateway_metadata(
    *,
    routing_decision: Any | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    include_trace: bool = True,
) -> dict[str, Any]:
    """Build a complete A2A ``metadata`` dict from all gateway extensions.

    Convenience function that combines routing, session, and trace
    context into a single metadata dict for ``adapter.send_message()``.

    Args:
        routing_decision: Optional ``RoutingDecision`` instance.
        session_id: Optional MCP session ID.
        context_id: Optional A2A context ID.
        include_trace: When ``True`` (default), injects the current OTel
            W3C trace context.

    Returns:
        A metadata dict with populated extension keys, ready for A2A messages.
    """
    metadata: dict[str, Any] = {}

    if routing_decision is not None:
        payload = pack_routing_metadata(routing_decision)
        if payload:
            metadata[ROUTING_METADATA_URI] = payload

    session = pack_mcp_session(session_id=session_id, context_id=context_id)
    if session:
        metadata[MCP_SESSION_URI] = session

    if include_trace:
        trace_ctx = current_trace_context()
        if trace_ctx:
            metadata[TRACE_CONTEXT_URI] = trace_ctx

    return metadata
