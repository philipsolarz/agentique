"""Session: the harness's view of one tracked run — a core Run plus work-product
semantics (the artifact ``kind`` it converges to, and the ``artifact`` once done).

The run itself — its lifecycle state, and the resumable ``Paused`` snapshot when
parked — lives on the core :class:`~agentique.core.scheduler.Scheduler` (addressed
by ``id``); the Session does *not* own dispatch or resume. It holds ``agent_id``
(the id the agent is registered under on the Scheduler), never the live Agent, so a
Session carries nothing un-serializable and resume routes by id.

``SessionRecord`` is the serializable projection the shared store persists — the
core run state plus the harness's work-product metadata (``kind`` via the artifact,
the pending ``question``, the produced ``artifact_id``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentique.artifact import Artifact
from agentique.core import Paused

type SessionState = Literal["running", "paused", "done", "failed"]


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """The serializable projection of a Session held in the shared store."""

    id: str
    agent_id: str
    state: SessionState
    question: str | None = None
    artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class Session:
    """A tracked run plus its work-product semantics.

    ``question`` is set only while ``paused``; ``artifact`` only once ``done``;
    ``error`` only when ``failed``. ``kind`` is the artifact kind a completed run
    converges to; ``agent_id`` is the Scheduler registration the run resumes against.
    """

    id: str
    agent_id: str
    kind: str
    state: SessionState
    question: str | None = None
    artifact: Artifact | None = None
    error: str | None = None
    derived_from: tuple[str, ...] = ()

    def record(self) -> SessionRecord:
        """Project to the serializable record the store persists."""
        return SessionRecord(
            id=self.id,
            agent_id=self.agent_id,
            state=self.state,
            question=self.question,
            artifact_id=self.artifact.id if self.artifact is not None else None,
        )


@dataclass(frozen=True, slots=True)
class PausedRun:
    """A persisted paused run: everything needed to resume it in a fresh process.

    Carries the core :class:`~agentique.core.result.Paused` snapshot plus the
    harness's resume metadata (the ``kind`` a completion converges to, the pending
    ``question``, and the provenance ``derived_from``). It deliberately holds
    ``agent_id`` — the Scheduler registration id — and **never** the live Agent (a
    live model client and workspace-bound tools cannot, and must not, be persisted);
    on restart the app re-registers the agent under the same id and the snapshot is
    re-paired with it.
    """

    run_id: str
    agent_id: str
    kind: str
    question: str | None
    derived_from: tuple[str, ...]
    paused: Paused
