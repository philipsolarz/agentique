"""Shared store: where the harness's work converges in state.

Built on the :class:`~agentique.core.memory.Memory` seam (defaulting to
:class:`~agentique.memory.InMemoryStore`), it persists ``Artifact``s and
``SessionRecord``s as JSON strings under namespaced keys, with a small index per
kind so they can be listed. This is the convergence point an async coordinator
will later build on: agents write results here as artifacts rather than into an
orchestrator's context. Live resume state (the ``Context``) is held in memory by
the Coordinator, not serialized here.
"""

from __future__ import annotations

import json

from agentique.code.artifact import Artifact
from agentique.code.session import SessionRecord
from agentique.core import Memory
from agentique.memory import InMemoryStore


def _artifact_json(artifact: Artifact) -> str:
    return json.dumps(
        {
            "id": artifact.id,
            "kind": artifact.kind,
            "payload": artifact.payload,
            "status": artifact.status,
        }
    )


def _artifact_from(raw: str) -> Artifact:
    data = json.loads(raw)
    return Artifact(
        id=data["id"],
        kind=data["kind"],
        payload=data["payload"],
        status=data["status"],
    )


def _session_json(record: SessionRecord) -> str:
    return json.dumps(
        {
            "id": record.id,
            "agent_name": record.agent_name,
            "state": record.state,
            "question": record.question,
            "artifact_id": record.artifact_id,
        }
    )


def _session_from(raw: str) -> SessionRecord:
    data = json.loads(raw)
    return SessionRecord(
        id=data["id"],
        agent_name=data["agent_name"],
        state=data["state"],
        question=data["question"],
        artifact_id=data["artifact_id"],
    )


class Store:
    """A durable-seam-backed store of artifacts and session records."""

    def __init__(self, memory: Memory | None = None) -> None:
        self._memory: Memory = memory if memory is not None else InMemoryStore()

    async def put_artifact(self, artifact: Artifact) -> None:
        await self._memory.set(f"artifact:{artifact.id}", _artifact_json(artifact))
        await self._extend_index("artifacts", artifact.id)

    async def get_artifact(self, artifact_id: str) -> Artifact | None:
        raw = await self._memory.get(f"artifact:{artifact_id}")
        return _artifact_from(raw) if raw is not None else None

    async def artifacts(self) -> tuple[Artifact, ...]:
        out: list[Artifact] = []
        for key in await self._index("artifacts"):
            artifact = await self.get_artifact(key)
            if artifact is not None:
                out.append(artifact)
        return tuple(out)

    async def put_session(self, record: SessionRecord) -> None:
        await self._memory.set(f"session:{record.id}", _session_json(record))
        await self._extend_index("sessions", record.id)

    async def get_session(self, session_id: str) -> SessionRecord | None:
        raw = await self._memory.get(f"session:{session_id}")
        return _session_from(raw) if raw is not None else None

    async def sessions(self) -> tuple[SessionRecord, ...]:
        out: list[SessionRecord] = []
        for key in await self._index("sessions"):
            record = await self.get_session(key)
            if record is not None:
                out.append(record)
        return tuple(out)

    async def _index(self, name: str) -> tuple[str, ...]:
        raw = await self._memory.get(f"index:{name}")
        return tuple(json.loads(raw)) if raw is not None else ()

    async def _extend_index(self, name: str, key: str) -> None:
        keys = list(await self._index(name))
        if key not in keys:
            keys.append(key)
            await self._memory.set(f"index:{name}", json.dumps(keys))
