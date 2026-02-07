"""Tests for gRPC transport and transport configuration."""

from __future__ import annotations

from typing import Any

import pytest

from agentique.adapters.a2a.client import A2AClientPool
from agentique.core.config import AgentiqueConfig


# ---- Config tests ----


def test_config_default_extensions():
    """AgentiqueConfig should have empty extensions by default."""
    config = AgentiqueConfig()
    assert config.extensions == []


def test_config_custom_extensions():
    """AgentiqueConfig should accept custom extensions."""
    config = AgentiqueConfig(
        extensions=["urn:a2a:ext:tracing", "urn:a2a:ext:custom"],
    )
    assert len(config.extensions) == 2
    assert "urn:a2a:ext:tracing" in config.extensions


def test_config_default_supported_transports():
    """AgentiqueConfig should have empty supported_transports by default."""
    config = AgentiqueConfig()
    assert config.supported_transports == []


def test_config_custom_transports():
    """AgentiqueConfig should accept custom supported_transports."""
    config = AgentiqueConfig(
        supported_transports=["JSONRPC", "GRPC"],
    )
    assert len(config.supported_transports) == 2
    assert "GRPC" in config.supported_transports


# ---- Client pool tests ----


def test_client_pool_default():
    """A2AClientPool should work with defaults."""
    pool = A2AClientPool()
    assert pool._extensions == []
    assert pool._supported_transports == []
    assert pool._grpc_channel_factory is None


def test_client_pool_with_extensions():
    """A2AClientPool should accept and store extensions."""
    pool = A2AClientPool(
        extensions=["urn:a2a:ext:tracing"],
    )
    assert pool._extensions == ["urn:a2a:ext:tracing"]


def test_client_pool_with_transports():
    """A2AClientPool should accept and store supported_transports."""
    pool = A2AClientPool(
        supported_transports=["JSONRPC", "GRPC"],
    )
    assert pool._supported_transports == ["JSONRPC", "GRPC"]


def test_client_pool_with_grpc_factory():
    """A2AClientPool should accept a gRPC channel factory."""
    def mock_factory(url: str) -> Any:
        return f"channel-{url}"

    pool = A2AClientPool(
        supported_transports=["GRPC"],
        grpc_channel_factory=mock_factory,
    )
    assert pool._grpc_channel_factory is mock_factory


def test_client_pool_extensions_and_transports():
    """A2AClientPool should support both extensions and transports."""
    pool = A2AClientPool(
        extensions=["urn:a2a:ext:tracing"],
        supported_transports=["JSONRPC", "GRPC"],
    )
    assert pool._extensions == ["urn:a2a:ext:tracing"]
    assert pool._supported_transports == ["JSONRPC", "GRPC"]


@pytest.mark.asyncio
async def test_client_pool_close_empty():
    """Closing an empty pool should not raise."""
    pool = A2AClientPool()
    await pool.close()


def test_client_pool_with_user_config():
    """A2AClientPool with user_config should store it for get()."""
    from a2a.client import ClientConfig
    import httpx

    http_client = httpx.AsyncClient(timeout=30.0)
    config = ClientConfig(httpx_client=http_client)

    pool = A2AClientPool(client_config=config)
    assert pool._user_config is config
