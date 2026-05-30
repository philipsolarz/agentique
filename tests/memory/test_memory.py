"""Both Memory implementations, tested against the same behavioral contract.

Parametrizing over both stores keeps them honest against one Protocol: anything
asserted here must hold for every conformant Memory. ``FileStore`` additionally
proves durability — a fresh instance over the same path sees prior writes.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from agentique.core.memory import Memory
from agentique.memory import FileStore, InMemoryStore

_Factory = Callable[[Path], Memory]

_FACTORIES: list[_Factory] = [
    lambda _tmp: InMemoryStore(),
    lambda tmp: FileStore(tmp / "mem.json"),
]


@pytest.fixture(params=_FACTORIES)
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Memory:
    factory: _Factory = request.param
    return factory(tmp_path)


async def test_get_missing_key_returns_none(store: Memory) -> None:
    assert await store.get("absent") is None


async def test_set_then_get_roundtrips(store: Memory) -> None:
    await store.set("k", "v")
    assert await store.get("k") == "v"


async def test_set_overwrites(store: Memory) -> None:
    await store.set("k", "first")
    await store.set("k", "second")
    assert await store.get("k") == "second"


async def test_file_store_is_durable_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "mem.json"
    await FileStore(path).set("k", "persisted")
    # a brand-new instance over the same path must see the earlier write.
    assert await FileStore(path).get("k") == "persisted"


def _is_memory(candidate: Memory) -> bool:
    return True


def test_both_satisfy_memory_protocol_statically(tmp_path: Path) -> None:
    # Passing each store where a ``Memory`` is annotated exercises the structural
    # check at type-check time; the runtime assert just keeps the test live.
    assert _is_memory(InMemoryStore())
    assert _is_memory(FileStore(tmp_path / "m.json"))
