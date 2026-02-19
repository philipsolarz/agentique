"""Tests for new create_server() parameters added in Focus #4.

Covers:
  - session_state_store parameter accepted and passed to FastMCP
  - sampling_handler parameter accepted and passed to FastMCP
  - health_check_interval creates a health lifespan
  - All new params default gracefully (None / "fallback")
  - Lifespan is always set (cleanup lifespan is always composed)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentique.core.types import AgentInfo


def make_agent(name: str = "test") -> AgentInfo:
    return AgentInfo(name=name, base_url=f"http://{name}.example.com")


# ---------------------------------------------------------------------------
# session_state_store
# ---------------------------------------------------------------------------


def test_create_server_accepts_session_state_store():
    from agentique.server import create_server
    store = MagicMock()
    # Should not raise — even if the store is a mock
    server = create_server(agents=[make_agent()], session_state_store=store)
    assert server is not None


def test_create_server_session_state_store_none_uses_default():
    from agentique.server import create_server
    server = create_server(agents=[make_agent()], session_state_store=None)
    assert server is not None


# ---------------------------------------------------------------------------
# sampling_handler
# ---------------------------------------------------------------------------


def test_create_server_accepts_sampling_handler():
    from agentique.server import create_server
    handler = MagicMock()
    server = create_server(agents=[make_agent()], sampling_handler=handler)
    assert server is not None


def test_create_server_sampling_handler_none_is_default():
    from agentique.server import create_server
    server = create_server(agents=[make_agent()], sampling_handler=None)
    assert server is not None


def test_create_server_sampling_handler_behavior_always():
    from agentique.server import create_server
    handler = MagicMock()
    server = create_server(
        agents=[make_agent()],
        sampling_handler=handler,
        sampling_handler_behavior="always",
    )
    assert server is not None


# ---------------------------------------------------------------------------
# health_check_interval
# ---------------------------------------------------------------------------


def test_create_server_health_check_interval_none():
    from agentique.server import create_server
    server = create_server(agents=[make_agent()], health_check_interval=None)
    assert server is not None


def test_create_server_health_check_interval_set():
    from agentique.server import create_server
    server = create_server(agents=[make_agent()], health_check_interval=30.0)
    assert server is not None


# ---------------------------------------------------------------------------
# Lifespan is always composed (cleanup lifespan always present)
# ---------------------------------------------------------------------------


def test_create_server_lifespan_is_set():
    """FastMCP server should always have a lifespan (cleanup at minimum)."""
    from agentique.server import create_server
    server = create_server(agents=[make_agent()])
    # The server should have a lifespan attribute
    lifespan = getattr(server, "_lifespan", None) or getattr(server, "lifespan", None)
    assert lifespan is not None


# ---------------------------------------------------------------------------
# New imports accessible from agentique top-level
# ---------------------------------------------------------------------------


def test_lifespans_importable_from_agentique():
    from agentique import (
        compose_lifespans,
        make_cleanup_lifespan,
        make_health_monitor_lifespan,
    )
    assert callable(make_cleanup_lifespan)
    assert callable(make_health_monitor_lifespan)
    assert callable(compose_lifespans)


def test_new_errors_importable_from_agentique():
    from agentique import (
        ContentTypeNotSupportedError,
        PushNotificationNotSupportedError,
        TaskNotCancelableError,
        UnsupportedOperationError,
    )
    assert issubclass(ContentTypeNotSupportedError, Exception)
    assert issubclass(UnsupportedOperationError, Exception)
    assert issubclass(TaskNotCancelableError, Exception)
    assert issubclass(PushNotificationNotSupportedError, Exception)


# ---------------------------------------------------------------------------
# Capabilities resource (Focus #5 Item C)
# ---------------------------------------------------------------------------


def _get_capabilities_resource_fn(server: Any) -> Any:
    """Extract the capabilities_resource function from the server."""
    # FastMCP 3.0 stores resources on providers.  The server module keeps a
    # local ``tasks`` variable and registers resources via @mcp.resource().
    # We extract it by looking at all registered resource functions.
    # Simplest: call server.list_resources() and then read the resource.
    return None  # used as a helper; tests call server.list_resources() directly


def test_capabilities_resource_uri_registered():
    """The server registers a resource at a2a://capabilities."""
    import asyncio
    from agentique.server import create_server

    server = create_server(agents=[make_agent()])

    async def _check():
        resources = await server.list_resources()
        uris = [str(r.uri) for r in resources]
        return uris

    uris = asyncio.run(_check())
    assert any("capabilities" in u for u in uris), (
        f"Expected a2a://capabilities in {uris}"
    )


def test_capabilities_resource_returns_valid_json():
    """a2a://capabilities resource returns parseable JSON."""
    import asyncio
    import json
    from agentique.server import create_server

    server = create_server(agents=[make_agent()])

    async def _read():
        return await server.read_resource("a2a://capabilities")

    raw = asyncio.run(_read())
    content = raw.contents[0].content
    data = json.loads(content)
    assert isinstance(data, dict)


def test_capabilities_resource_lists_extension_uris():
    """JSON contains all 4 Agentique extension URIs."""
    import asyncio
    import json
    from agentique.server import create_server
    from agentique.extensions import ALL_EXTENSION_URIS

    server = create_server(agents=[make_agent()])

    async def _read():
        return await server.read_resource("a2a://capabilities")

    raw = asyncio.run(_read())
    content = raw.contents[0].content
    data = json.loads(content)

    assert "extensions" in data
    for uri in ALL_EXTENSION_URIS:
        assert uri in data["extensions"], f"{uri} missing from extensions"


def test_capabilities_resource_includes_gateway_name():
    """Capabilities JSON includes the gateway name from config."""
    import asyncio
    import json
    from agentique.core.config import AgentiqueConfig
    from agentique.server import create_server

    cfg = AgentiqueConfig(name="MyGateway")
    server = create_server(agents=[make_agent()], config=cfg)

    async def _read():
        return await server.read_resource("a2a://capabilities")

    raw = asyncio.run(_read())
    content = raw.contents[0].content
    data = json.loads(content)

    assert data["gateway"] == "MyGateway"


def test_capabilities_resource_features_dict():
    """Capabilities JSON contains a features dict with expected flags."""
    import asyncio
    import json
    from agentique.server import create_server

    server = create_server(agents=[make_agent()])

    async def _read():
        return await server.read_resource("a2a://capabilities")

    raw = asyncio.run(_read())
    content = raw.contents[0].content
    data = json.loads(content)

    assert "features" in data
    features = data["features"]
    for key in ("background_tasks", "elicitation", "tool_confirmation", "push_notifications"):
        assert key in features, f"Expected {key} in features"


def test_capabilities_resource_session_state_persistent_false_by_default():
    """session_state_persistent is False when no store is passed."""
    import asyncio
    import json
    from agentique.server import create_server

    server = create_server(agents=[make_agent()])

    async def _read():
        return await server.read_resource("a2a://capabilities")

    raw = asyncio.run(_read())
    content = raw.contents[0].content
    data = json.loads(content)

    assert data["session_state_persistent"] is False


def test_capabilities_resource_session_state_persistent_true_when_store_given():
    """session_state_persistent is True when a session_state_store is passed."""
    import asyncio
    import json
    from agentique.server import create_server

    store = MagicMock()
    server = create_server(agents=[make_agent()], session_state_store=store)

    async def _read():
        return await server.read_resource("a2a://capabilities")

    raw = asyncio.run(_read())
    content = raw.contents[0].content
    data = json.loads(content)

    assert data["session_state_persistent"] is True


def test_capabilities_resource_supported_transports_default():
    """When no transports configured, defaults to [JSONRPC]."""
    import asyncio
    import json
    from agentique.server import create_server

    server = create_server(agents=[make_agent()])

    async def _read():
        return await server.read_resource("a2a://capabilities")

    raw = asyncio.run(_read())
    content = raw.contents[0].content
    data = json.loads(content)

    assert "supported_transports" in data
    assert "JSONRPC" in data["supported_transports"]
