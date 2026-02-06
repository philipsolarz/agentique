"""Async event emitter for agentique lifecycle hooks.

Supports both sync and async handlers. Events are defined as simple
string names; handlers are plain callables.

Lifecycle events emitted by agentique:
    - ``server.start`` / ``server.stop``
    - ``agent.discovered`` / ``agent.lost``
    - ``tool.called`` / ``tool.completed`` / ``tool.failed``
    - ``task.created`` / ``task.state_changed`` / ``task.completed``
    - ``message.sent`` / ``message.received``
    - ``stream.chunk``
    - ``error``
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

EventHook = Callable[..., Any] | Callable[..., Coroutine[Any, Any, Any]]


class AsyncEventEmitter:
    """Simple async event emitter with support for sync and async handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHook]] = defaultdict(list)

    def on(self, event: str, handler: EventHook) -> None:
        """Register *handler* for *event*."""
        self._handlers[event].append(handler)

    def off(self, event: str, handler: EventHook) -> None:
        """Remove *handler* from *event*."""
        handlers = self._handlers.get(event)
        if handlers:
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    async def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Emit *event*, calling all registered handlers concurrently."""
        handlers = self._handlers.get(event, [])
        if not handlers:
            return

        tasks: list[Coroutine[Any, Any, Any]] = []
        for handler in handlers:
            try:
                result = handler(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    tasks.append(result)
            except Exception:
                logger.exception("Sync event handler for '%s' raised", event)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.exception(
                        "Async event handler for '%s' raised: %s", event, r,
                    )

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
