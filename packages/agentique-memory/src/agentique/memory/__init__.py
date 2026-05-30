"""Agentique memory: durable, cross-run :class:`~agentique.core.memory.Memory`
implementations.

Two concretes satisfy the same Protocol structurally:

* :class:`InMemoryStore` — ephemeral process-lifetime storage (a dict), useful
  for tests and short-lived runs.
* :class:`FileStore` — durable storage backed by a JSON file on disk.

Both depend only on ``agentique-core`` contracts and the standard library.
"""

from agentique.memory.file_store import FileStore
from agentique.memory.in_memory import InMemoryStore

__all__ = ["FileStore", "InMemoryStore"]
