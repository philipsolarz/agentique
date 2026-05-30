"""An in-process Memory: a dict that lives for the duration of the process.

Satisfies the :class:`~agentique.core.memory.Memory` Protocol structurally. Not
durable across runs despite implementing the durable seam — it is the simplest
conformant store, ideal for tests and for composing higher-level behavior
without touching disk.
"""

from __future__ import annotations


class InMemoryStore:
    """A ``Memory`` backed by an in-process dictionary."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(self, key: str, value: str) -> None:
        self._data[key] = value
