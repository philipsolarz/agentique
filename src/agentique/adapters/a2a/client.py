"""A2A SDK client pool.

Manages creation, caching, and lifecycle of ``a2a-sdk`` client instances.
"""

from __future__ import annotations

from collections.abc import Sequence
import inspect
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

try:
    from a2a.client import ClientConfig, ClientFactory
    from a2a.types import PushNotificationConfig
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
        supported_transports: Sequence[str] | None = None,
        use_client_preference: bool = False,
        extensions: Sequence[str] | None = None,
        push_notification_configs: Sequence[PushNotificationConfig] | None = None,
        resolver_http_kwargs: dict[str, Any] | None = None,
        extra_transports: dict[str, Any] | None = None,
    ) -> None:
        self._timeout = timeout
        self._user_config = client_config
        self._card_path = card_path
        self._supported_transports = list(supported_transports or [])
        self._use_client_preference = use_client_preference
        self._extensions = list(extensions or [])
        self._push_notification_configs = list(push_notification_configs or [])
        self._resolver_http_kwargs = dict(resolver_http_kwargs or {})
        self._extra_transports = dict(extra_transports or {})
        self._clients: dict[str, Any] = {}

    async def get(self, base_url: str) -> Any:
        """Get or create a client for *base_url*."""
        if base_url in self._clients:
            return self._clients[base_url]

        config = self._build_client_config()

        kwargs: dict[str, Any] = {
            "client_config": config,
        }
        if self._card_path:
            kwargs["relative_card_path"] = self._card_path
        if self._resolver_http_kwargs:
            kwargs["resolver_http_kwargs"] = dict(self._resolver_http_kwargs)
        if self._extra_transports:
            kwargs["extra_transports"] = dict(self._extra_transports)
        if self._extensions:
            kwargs["extensions"] = list(self._extensions)

        client = await ClientFactory.connect(base_url, **kwargs)
        if client is None:
            raise RuntimeError(f"Failed to connect to A2A agent at {base_url}")

        self._clients[base_url] = client
        return client

    def _build_client_config(self) -> ClientConfig:
        if self._user_config is not None:
            return self._user_config

        http_client = httpx.AsyncClient(timeout=self._timeout)
        kwargs: dict[str, Any] = {
            "httpx_client": http_client,
            "use_client_preference": self._use_client_preference,
        }
        if self._supported_transports:
            kwargs["supported_transports"] = list(self._supported_transports)
        if self._extensions:
            kwargs["extensions"] = list(self._extensions)
        if self._push_notification_configs:
            kwargs["push_notification_configs"] = list(
                self._push_notification_configs,
            )

        return ClientConfig(**kwargs)

    async def close(self) -> None:
        """Close all cached clients."""
        for client in self._clients.values():
            close_fn = getattr(client, "aclose", None) or getattr(client, "close", None)
            if callable(close_fn):
                result = close_fn()
                if inspect.isawaitable(result):
                    await result
        self._clients.clear()
