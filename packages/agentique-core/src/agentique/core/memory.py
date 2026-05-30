"""The Memory seam: durable, cross-run, possibly shared state.

Distinct from :class:`~agentique.core.context.Context` (ephemeral, single-run)
on purpose: keeping them separate types makes it a type error to persist
ephemeral working state by accident. The durable/ephemeral split is committed
and pressure-tested at A5, where the concrete in-memory and file-backed
implementations also arrive.

Async because a real backing store (a file, a database) performs I/O.
"""

from __future__ import annotations

from typing import Protocol


class Memory(Protocol):
    """A durable key/value store that outlives a single run."""

    async def get(self, key: str) -> str | None:
        """Return the stored value for ``key``, or ``None`` if absent."""
        ...

    async def set(self, key: str, value: str) -> None:
        """Durably associate ``value`` with ``key``."""
        ...
