"""Tests for the adapter registry."""

from __future__ import annotations

import pytest

from agentique.core.registry import (
    _REGISTRY,
    create_adapter,
    discover_adapters,
    list_protocols,
    register_adapter,
)
from agentique.core.types import AgentInfo


def test_built_in_adapters_registered():
    """Built-in A2A and HTTP adapters should be auto-registered."""
    # Force import of the adapters module to trigger registration
    import agentique.adapters  # noqa: F401

    adapters = discover_adapters()
    assert "a2a" in adapters
    assert "http" in adapters


def test_list_protocols():
    import agentique.adapters  # noqa: F401
    protocols = list_protocols()
    assert "a2a" in protocols
    assert "http" in protocols
    assert protocols == sorted(protocols)


def test_register_decorator():
    @register_adapter("test_proto")
    class TestFactory:
        protocol_name = "test_proto"
        def create(self, agents, **kwargs):
            return None

    assert "test_proto" in _REGISTRY
    # Cleanup
    del _REGISTRY["test_proto"]


def test_create_adapter_unknown_protocol():
    with pytest.raises(ValueError, match="No adapter registered"):
        create_adapter("nonexistent_protocol", {})


def test_create_adapter_http():
    """Creating an HTTP adapter should return an HttpAgentAdapter."""
    import agentique.adapters  # noqa: F401

    agents = {"test": AgentInfo(name="test", base_url="http://localhost:8080")}
    adapter = create_adapter("http", agents)

    from agentique.adapters.http.adapter import HttpAgentAdapter
    assert isinstance(adapter, HttpAgentAdapter)
