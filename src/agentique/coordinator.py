"""Coordinator: owns the live Sessions and the shared store, and drives runs.

It *uses* ``Runtime``s — it is **not** a ``Runtime`` subclass: a Runtime drives
one run, the Coordinator coordinates many. ``dispatch`` launches a run as a new
Session and maps its terminal ``Result`` into stored state (a ``proposed``
Artifact on completion); ``resume`` routes a human's answer to a *specific* paused
Session by id and continues that run. Full async fan-out (many sessions in flight
at once) builds on this spine — this session exercises it one run at a time.

It also holds a registry of named :class:`~agentique.role.Role`\\ s, so an
application (or an orchestrator) can dispatch a specialist *by name*
(``dispatch_role``) — an explicit coordination act, not a side effect hidden in a
tool. Artifact promotion is likewise explicit: ``promote_artifact`` moves a stored
artifact to any application-defined status, with ``approve``/``reject`` as the
common cases.
"""

from __future__ import annotations

from dataclasses import replace

from agentique.artifact import Artifact
from agentique.core import Agent, Blocked, Completed, NeedsHuman, Result, Runtime
from agentique.role import Role
from agentique.session import Session
from agentique.store import Store


class Coordinator:
    """Owns live Sessions and the shared store; launches and resumes runs."""

    def __init__(
        self, store: Store | None = None, runtime: Runtime | None = None
    ) -> None:
        self._store: Store = store if store is not None else Store()
        self._runtime: Runtime = runtime if runtime is not None else Runtime()
        self._sessions: dict[str, Session] = {}
        self._roles: dict[str, Role] = {}
        self._session_seq = 0
        self._artifact_seq = 0

    @property
    def store(self) -> Store:
        return self._store

    def session(self, session_id: str) -> Session | None:
        """The live Session for ``session_id``, or ``None`` if there is none."""
        return self._sessions.get(session_id)

    def sessions(self) -> tuple[Session, ...]:
        """All live Sessions, in creation order."""
        return tuple(self._sessions.values())

    def register_role(self, role: Role) -> None:
        """Register a named specialist that ``dispatch_role`` can launch."""
        self._roles[role.name] = role

    def role(self, name: str) -> Role | None:
        """The registered Role named ``name``, or ``None`` if there is none."""
        return self._roles.get(name)

    def roles(self) -> tuple[Role, ...]:
        """All registered Roles, in registration order."""
        return tuple(self._roles.values())

    async def dispatch(
        self, agent: Agent, prompt: str, *, kind: str = "result"
    ) -> Session:
        """Launch ``agent`` on ``prompt`` as a new Session and store its outcome."""
        self._session_seq += 1
        session = Session(
            id=f"s{self._session_seq}", agent=agent, kind=kind, state="running"
        )
        self._sessions[session.id] = session
        result = await self._runtime.run(agent, prompt)
        return await self._finalize(session, result)

    async def dispatch_role(self, name: str, prompt: str) -> Session:
        """Dispatch the registered Role ``name`` on ``prompt`` as a new Session."""
        role = self._roles.get(name)
        if role is None:
            raise KeyError(f"no role {name!r}")
        return await self.dispatch(role.agent, prompt, kind=role.kind)

    async def resume(self, session_id: str, answer: str) -> Session:
        """Continue the paused Session ``session_id`` with the human's ``answer``."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"no session {session_id!r}")
        if session.state != "paused" or session.paused is None:
            raise ValueError(f"session {session_id!r} is not paused")
        result = await self._runtime.resume(session.agent, session.paused, answer)
        return await self._finalize(session, result)

    async def promote_artifact(self, artifact_id: str, status: str) -> Artifact:
        """Move a stored artifact to ``status`` and persist it.

        The general, explicit promotion act an operator drives. ``status`` is an
        application-defined string; the harness assigns it no meaning.
        """
        artifact = await self._store.get_artifact(artifact_id)
        if artifact is None:
            raise KeyError(f"no artifact {artifact_id!r}")
        updated = artifact.with_status(status)
        await self._store.put_artifact(updated)
        return updated

    async def approve_artifact(self, artifact_id: str) -> Artifact:
        """Mark a stored artifact ``approved`` and persist it (common case)."""
        return await self.promote_artifact(artifact_id, "approved")

    async def reject_artifact(self, artifact_id: str) -> Artifact:
        """Mark a stored artifact ``rejected`` and persist it (common case)."""
        return await self.promote_artifact(artifact_id, "rejected")

    async def _finalize(self, session: Session, result: Result) -> Session:
        """Map a run's terminal Result into the Session and the shared store."""
        match result:
            case Completed(output=output):
                self._artifact_seq += 1
                artifact = Artifact(
                    id=f"a{self._artifact_seq}", kind=session.kind, payload=output
                )
                await self._store.put_artifact(artifact)
                session = replace(
                    session,
                    state="done",
                    artifact=artifact,
                    paused=None,
                    question=None,
                )
            case NeedsHuman(question=question, paused=paused):
                session = replace(
                    session, state="paused", paused=paused, question=question
                )
            case Blocked(reason=reason):
                session = replace(
                    session, state="failed", error=reason, paused=None, question=None
                )
        self._sessions[session.id] = session
        await self._store.put_session(session.record())
        return session
