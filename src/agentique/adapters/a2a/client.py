"""A2A SDK client pool.

Manages creation, caching, and lifecycle of ``a2a-sdk`` client instances.

Supports:
    - Per-client extension propagation via ``ClientConfig.extensions``
    - Transport selection via ``ClientConfig.supported_transports``
    - Optional gRPC channel factory for gRPC transport
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

try:
    from a2a.client import ClientConfig, ClientFactory
except ImportError as exc:
    raise RuntimeError(
        "a2a-sdk is required for the A2A adapter. "
        "Install with: pip install 'agentique[a2a]'"
    ) from exc


class A2AClientPool:
    """Creates, caches, and manages A2A SDK clients by base URL.

    Args:
        timeout: HTTP request timeout in seconds.
        client_config: Override the SDK's ``ClientConfig``.
        card_path: Custom relative path for agent card discovery.
        extensions: A2A extension URIs to advertise to agents.
        supported_transports: Ordered list of preferred transports
            (e.g. ``["JSONRPC", "GRPC"]``).  Empty means JSON-RPC only.
        grpc_channel_factory: Callable that creates a gRPC ``Channel``
            from a URL string.  Required when ``"GRPC"`` is listed
            in *supported_transports*.
    """

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        client_config: ClientConfig | None = None,
        card_path: str | None = None,
        extensions: list[str] | None = None,
        supported_transports: list[str] | None = None,
        grpc_channel_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._timeout = timeout
        self._user_config = client_config
        self._card_path = card_path
        self._extensions = extensions or []
        self._supported_transports = supported_transports or []
        self._grpc_channel_factory = grpc_channel_factory
        self._clients: dict[str, Any] = {}

    async def get(self, base_url: str) -> Any:
        """Get or create a client for *base_url*."""
        if base_url in self._clients:
            return self._clients[base_url]

        config = self._user_config
        if config is None:
            http_client = httpx.AsyncClient(timeout=self._timeout)
            config_kwargs: dict[str, Any] = {"httpx_client": http_client}

            # Extensions
            if self._extensions:
                config_kwargs["extensions"] = list(self._extensions)

            # Transport selection
            if self._supported_transports:
                config_kwargs["supported_transports"] = list(
                    self._supported_transports
                )

            # gRPC channel factory
            if self._grpc_channel_factory is not None:
                config_kwargs["grpc_channel_factory"] = self._grpc_channel_factory

            config = ClientConfig(**config_kwargs)

        kwargs: dict[str, Any] = {"client_config": config}
        if self._card_path:
            kwargs["relative_card_path"] = self._card_path

        client = await ClientFactory.connect(base_url, **kwargs)
        if client is None:
            raise RuntimeError(f"Failed to connect to A2A agent at {base_url}")

        self._clients[base_url] = client
        return client

    async def close(self) -> None:
        """Close all cached clients."""
        for client in self._clients.values():
            close_fn = getattr(client, "aclose", None) or getattr(client, "close", None)
            if callable(close_fn):
                result = close_fn()
                if inspect.isawaitable(result):
                    await result
        self._clients.clear()
