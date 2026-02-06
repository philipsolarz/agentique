"""A2A SDK client pool.

Manages creation, caching, and lifecycle of ``a2a-sdk`` client instances.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

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
    """Creates, caches, and manages A2A SDK clients by base URL."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        client_config: ClientConfig | None = None,
        card_path: str | None = None,
    ) -> None:
        self._timeout = timeout
        self._user_config = client_config
        self._card_path = card_path
        self._clients: dict[str, Any] = {}

    async def get(self, base_url: str) -> Any:
        """Get or create a client for *base_url*."""
        if base_url in self._clients:
            return self._clients[base_url]

        config = self._user_config
        if config is None:
            http_client = httpx.AsyncClient(timeout=self._timeout)
            config = ClientConfig(httpx_client=http_client)

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
