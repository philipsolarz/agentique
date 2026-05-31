"""Coordinator: the harness's operator layer over the core Scheduler.

Dispatch, run-tracking, run-lineage, and resume-routing live in the core
:class:`~agentique.core.scheduler.Scheduler`; the Coordinator is the thin harness
adapter that maps a run's terminal :class:`~agentique.core.result.Result` into the
*work-product* layer it owns — a ``proposed`` :class:`~agentique.artifact.Artifact`
on completion, persisted in the shared :class:`~agentique.store.Store` — and keeps a
:class:`~agentique.session.Session` projection of each run. It also holds the
registry of named :class:`~agentique.role.Role`\\ s (dispatch a specialist *by
name*) and the explicit artifact-promotion verbs (``approve``/``reject``/
``promote_artifact``). It holds **no** copy of run state: ``state``/``paused`` are
read from the Scheduler's :class:`~agentique.core.scheduler.Run`.

The Session id *is* the Scheduler run id, so ``resume(session_id, …)`` routes
straight to ``Scheduler.resume(run_id, …)``.
"""

from __future__ import annotations

from agentique.artifact import Artifact
from agentique.core import (
    Agent,
    Blocked,
    Completed,
    NeedsHuman,
    Result,
    Run,
    Scheduler,
)
from agentique.role import Role
from agentique.session import PausedRun, Session
from agentique.store import Store


class Coordinator:
    """Maps Scheduler runs into the harness's artifact/session work-product layer."""

    def __init__(
        self, store: Store | None = None, scheduler: Scheduler | None = None
    ) -> None:
        self._store: Store = store if store is not None else Store()
        self._scheduler: Scheduler = scheduler if scheduler is not None else Scheduler()
        self._sessions: dict[str, Session] = {}
        self._roles: dict[str, Role] = {}
        self._agent_seq = 0
        self._artifact_seq = 0

    @property
    def store(self) -> Store:
        return self._store

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    def session(self, session_id: str) -> Session | None:
        """The Session projection for ``session_id``, or ``None``."""
        return self._sessions.get(session_id)

    def sessions(self) -> tuple[Session, ...]:
        """All tracked Sessions, in creation order."""
        return tuple(self._sessions.values())

    def register_role(self, role: Role) -> None:
        """Register a named specialist and its agent (under ``role.name``) so
        ``dispatch_role`` can launch it and a paused run can resume against it."""
        self._roles[role.name] = role
        self._scheduler.register(role.name, role.agent)

    def role(self, name: str) -> Role | None:
        return self._roles.get(name)

    def roles(self) -> tuple[Role, ...]:
        return tuple(self._roles.values())

    async def dispatch(
        self,
        agent: Agent,
        prompt: str,
        *,
        kind: str = "result",
        derived_from: tuple[str, ...] = (),
    ) -> Session:
        """Launch ``agent`` on ``prompt`` as a new run and project its outcome.

        The ad-hoc agent is registered on the Scheduler under a fresh id; for a
        named specialist use :meth:`dispatch_role`. ``derived_from`` records the
        provenance edges (ids of artifacts this run builds on) on the produced
        artifact — supplied by the caller, assigned no meaning by the harness.
        """
        self._agent_seq += 1
        agent_id = f"agent-{self._agent_seq}"
        self._scheduler.register(agent_id, agent)
        return await self._launch(agent_id, prompt, kind, derived_from)

    async def dispatch_role(
        self, name: str, prompt: str, *, derived_from: tuple[str, ...] = ()
    ) -> Session:
        """Dispatch the registered Role ``name`` on ``prompt`` as a new run."""
        role = self._roles.get(name)
        if role is None:
            raise KeyError(f"no role {name!r}")
        return await self._launch(role.name, prompt, role.kind, derived_from)

    async def load_paused_runs(self) -> tuple[Session, ...]:
        """Restore persisted paused runs into this Coordinator and its Scheduler.

        For a fresh process: the app re-registers its agents on the Scheduler under
        the same ids, then calls this; each persisted :class:`PausedRun` is re-seated
        on the Scheduler and re-projected as a paused Session, so ``resume(id, …)``
        continues it. (The app owns re-registration because only it can rebuild the
        live, workspace-bound agents — the harness persists run state, never agents.)
        """
        restored: list[Session] = []
        for record in await self._store.paused_runs():
            self._scheduler.restore(
                Run(
                    id=record.run_id,
                    agent_id=record.agent_id,
                    state="paused",
                    paused=record.paused,
                )
            )
            session = Session(
                id=record.run_id,
                agent_id=record.agent_id,
                kind=record.kind,
                state="paused",
                question=record.question,
                derived_from=record.derived_from,
            )
            self._sessions[session.id] = session
            restored.append(session)
        return tuple(restored)

    async def resume(self, session_id: str, answer: str) -> Session:
        """Continue the paused run ``session_id`` with the human's ``answer``."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"no session {session_id!r}")
        if session.state != "paused":
            raise ValueError(f"session {session_id!r} is not paused")
        result = await self._scheduler.resume(session_id, answer)
        run = self._scheduler.run(session_id)
        assert run is not None  # the run was just resumed
        return await self._finalize(
            run, session.agent_id, session.kind, result, session.derived_from
        )

    async def promote_artifact(self, artifact_id: str, status: str) -> Artifact:
        """Move a stored artifact to ``status`` and persist it (status is an
        application-defined string the harness assigns no meaning to)."""
        artifact = await self._store.get_artifact(artifact_id)
        if artifact is None:
            raise KeyError(f"no artifact {artifact_id!r}")
        updated = artifact.with_status(status)
        await self._store.put_artifact(updated)
        return updated

    async def approve_artifact(self, artifact_id: str) -> Artifact:
        return await self.promote_artifact(artifact_id, "approved")

    async def reject_artifact(self, artifact_id: str) -> Artifact:
        return await self.promote_artifact(artifact_id, "rejected")

    async def _launch(
        self, agent_id: str, prompt: str, kind: str, derived_from: tuple[str, ...]
    ) -> Session:
        dispatched = await self._scheduler.dispatch(agent_id, prompt)
        run = self._scheduler.run(dispatched.run_id)
        assert run is not None  # the dispatch just recorded this run
        return await self._finalize(
            run, agent_id, kind, dispatched.result, derived_from
        )

    async def _finalize(
        self,
        run: Run,
        agent_id: str,
        kind: str,
        result: Result,
        derived_from: tuple[str, ...],
    ) -> Session:
        """Map a run's terminal Result into the Session projection and the store."""
        match result:
            case Completed(output=output):
                self._artifact_seq += 1
                artifact = Artifact(
                    id=f"a{self._artifact_seq}",
                    kind=kind,
                    payload=self._store.build_payload(kind, output),
                    derived_from=derived_from,
                )
                await self._store.put_artifact(artifact)
                await self._store.delete_paused_run(run.id)
                session = Session(
                    id=run.id,
                    agent_id=agent_id,
                    kind=kind,
                    state="done",
                    artifact=artifact,
                    derived_from=derived_from,
                )
            case NeedsHuman(question=question):
                assert run.paused is not None  # a paused run carries its snapshot
                await self._store.put_paused_run(
                    PausedRun(
                        run_id=run.id,
                        agent_id=agent_id,
                        kind=kind,
                        question=question,
                        derived_from=derived_from,
                        paused=run.paused,
                    )
                )
                session = Session(
                    id=run.id,
                    agent_id=agent_id,
                    kind=kind,
                    state="paused",
                    question=question,
                    derived_from=derived_from,
                )
            case Blocked(reason=reason):
                await self._store.delete_paused_run(run.id)
                session = Session(
                    id=run.id,
                    agent_id=agent_id,
                    kind=kind,
                    state="failed",
                    error=reason,
                    derived_from=derived_from,
                )
        self._sessions[session.id] = session
        await self._store.put_session(session.record())
        return session
