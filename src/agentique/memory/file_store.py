"""A durable Memory backed by a single JSON file.

Satisfies the :class:`~agentique.core.memory.Memory` Protocol. The whole store is
a flat ``{str: str}`` object serialized to one JSON file; each ``set`` rewrites
it atomically (write to a temp file, then replace) so a crash mid-write cannot
corrupt the store.

The file I/O is synchronous, run off the event loop via ``asyncio.to_thread`` so
the async ``Memory`` contract is honored without blocking other tasks. This is a
correct, modest reference implementation — not a high-throughput store; a real
high-concurrency backend would use a database behind the same Protocol.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path


class FileStore:
    """A ``Memory`` persisted to a JSON file at ``path``."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    def _load(self) -> dict[str, str]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        data: dict[str, str] = json.loads(raw)
        return data

    def _store(self, data: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self._path)

    async def get(self, key: str) -> str | None:
        data = await asyncio.to_thread(self._load)
        return data.get(key)

    async def set(self, key: str, value: str) -> None:
        def _mutate() -> None:
            data = self._load()
            data[key] = value
            self._store(data)

        await asyncio.to_thread(_mutate)

    async def delete(self, key: str) -> None:
        def _mutate() -> None:
            data = self._load()
            if data.pop(key, None) is not None:  # absent key is a no-op
                self._store(data)

        await asyncio.to_thread(_mutate)
