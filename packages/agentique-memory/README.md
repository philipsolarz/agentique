# agentique-memory

Durable, cross-run `Memory` implementations for Agentique.

```python
from agentique.memory import InMemoryStore, FileStore
```

- **`InMemoryStore()`** — a process-lifetime dict; ideal for tests.
- **`FileStore(path)`** — durable storage backed by a single JSON file, written
  atomically; file I/O runs off the event loop via `asyncio.to_thread`.

Both satisfy the `agentique.core.Memory` Protocol (`async get`/`set`). Depends
only on `agentique-core` and the standard library.
