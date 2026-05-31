"""Shared store: where the harness's work converges in state.

Built on the :class:`~agentique.core.memory.Memory` seam (defaulting to
:class:`~agentique.memory.InMemoryStore`), it persists ``Artifact``s and
``SessionRecord``s as JSON under namespaced keys, with a small index per kind so
they can be listed.

Typed payloads round-trip via Pydantic: an artifact's payload is dumped with
``model_dump_json`` and rehydrated against an **app-supplied kind→model registry**
injected at construction, so the harness can reconstruct the concrete payload type
without ever learning what a ``"change"`` *is* — the concrete models live in the
application. An unregistered (or registry-free) kind falls back to the generic
:class:`~agentique.artifact.TextPayload`, so a bare ``Store()`` still works.
:meth:`build_payload` is the single place that turns a run's output text into a
typed payload, keeping the registry in one place for both write and read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from pydantic import BaseModel, TypeAdapter

from agentique.artifact import Artifact, TextPayload
from agentique.core import Memory, Paused
from agentique.memory import InMemoryStore
from agentique.session import PausedRun, SessionRecord

type KindRegistry = Mapping[str, type[BaseModel]]
"""Maps an artifact ``kind`` to the concrete Pydantic payload model for it. Defined
and supplied by the application; the harness only looks names up in it."""

# The core Paused snapshot is a (Pydantic) value, so it serializes for free.
_PAUSED = TypeAdapter(Paused)


def _session_json(record: SessionRecord) -> str:
    return json.dumps(
        {
            "id": record.id,
            "agent_id": record.agent_id,
            "state": record.state,
            "question": record.question,
            "artifact_id": record.artifact_id,
        }
    )


def _session_from(raw: str) -> SessionRecord:
    data = json.loads(raw)
    return SessionRecord(
        id=data["id"],
        agent_id=data["agent_id"],
        state=data["state"],
        question=data["question"],
        artifact_id=data["artifact_id"],
    )


class Store:
    """A durable-seam-backed store of artifacts and session records."""

    def __init__(
        self,
        memory: Memory | None = None,
        *,
        payload_models: KindRegistry | None = None,
    ) -> None:
        self._memory: Memory = memory if memory is not None else InMemoryStore()
        self._payload_models: dict[str, type[BaseModel]] = (
            dict(payload_models) if payload_models is not None else {}
        )

    def payload_model(self, kind: str) -> type[BaseModel]:
        """The payload model for ``kind`` — the app-registered one, else
        :class:`~agentique.artifact.TextPayload`."""
        return self._payload_models.get(kind, TextPayload)

    def build_payload(self, kind: str, output: str) -> BaseModel:
        """Turn a completed run's ``output`` text into the typed payload for
        ``kind``: parse it as the registered model, or wrap it as ``TextPayload``."""
        model = self._payload_models.get(kind)
        if model is None:
            return TextPayload(text=output)
        return model.model_validate_json(output)

    def _artifact_json(self, artifact: Artifact) -> str:
        return json.dumps(
            {
                "id": artifact.id,
                "kind": artifact.kind,
                "payload": artifact.payload.model_dump(mode="json"),
                "status": artifact.status,
                "derived_from": list(artifact.derived_from),
            }
        )

    def _artifact_from(self, raw: str) -> Artifact:
        data = json.loads(raw)
        payload = self.payload_model(data["kind"]).model_validate(data["payload"])
        return Artifact(
            id=data["id"],
            kind=data["kind"],
            payload=payload,
            status=data["status"],
            derived_from=tuple(data.get("derived_from", ())),
        )

    async def put_artifact(self, artifact: Artifact) -> None:
        await self._memory.set(f"artifact:{artifact.id}", self._artifact_json(artifact))
        await self._extend_index("artifacts", artifact.id)

    async def get_artifact(self, artifact_id: str) -> Artifact | None:
        raw = await self._memory.get(f"artifact:{artifact_id}")
        return self._artifact_from(raw) if raw is not None else None

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

    async def put_paused_run(self, record: PausedRun) -> None:
        """Persist a paused run so an approval gate survives a process restart.
        Stores the Paused snapshot + resume metadata only — never the live Agent."""
        data = json.dumps(
            {
                "run_id": record.run_id,
                "agent_id": record.agent_id,
                "kind": record.kind,
                "question": record.question,
                "derived_from": list(record.derived_from),
                "paused": json.loads(_PAUSED.dump_json(record.paused)),
            }
        )
        await self._memory.set(f"pausedrun:{record.run_id}", data)
        await self._extend_index("pausedruns", record.run_id)

    async def get_paused_run(self, run_id: str) -> PausedRun | None:
        raw = await self._memory.get(f"pausedrun:{run_id}")
        if raw is None:
            return None
        data = json.loads(raw)
        return PausedRun(
            run_id=data["run_id"],
            agent_id=data["agent_id"],
            kind=data["kind"],
            question=data["question"],
            derived_from=tuple(data["derived_from"]),
            paused=_PAUSED.validate_python(data["paused"]),
        )

    async def paused_runs(self) -> tuple[PausedRun, ...]:
        out: list[PausedRun] = []
        for key in await self._index("pausedruns"):
            record = await self.get_paused_run(key)
            if record is not None:
                out.append(record)
        return tuple(out)

    async def delete_paused_run(self, run_id: str) -> None:
        """Drop a persisted paused run once it has resolved (resumed to done/failed).
        A no-op if there is none."""
        if await self._memory.get(f"pausedrun:{run_id}") is None:
            return
        await self._memory.delete(f"pausedrun:{run_id}")
        keys = [k for k in await self._index("pausedruns") if k != run_id]
        await self._memory.set("index:pausedruns", json.dumps(keys))

    async def _index(self, name: str) -> tuple[str, ...]:
        raw = await self._memory.get(f"index:{name}")
        return tuple(json.loads(raw)) if raw is not None else ()

    async def _extend_index(self, name: str, key: str) -> None:
        keys = list(await self._index(name))
        if key not in keys:
            keys.append(key)
            await self._memory.set(f"index:{name}", json.dumps(keys))
