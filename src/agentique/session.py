"""Session: one running agent task the harness tracks.

A Session is the unit per-task pause acts on: a single ``Runtime`` run with an id,
a lifecycle state, and — when paused in ``NeedsHuman`` — the resumable ``Paused``
snapshot needed to continue *that* run. It bundles the live ``Agent`` with its
paused state so the Coordinator can resume one session by id while others are
untouched. It is a value (frozen); the Coordinator advances it with ``replace``.

``SessionRecord`` is the serializable projection the shared store persists — no
live ``Agent`` or ``Context``, just the metadata work converges on. (Resume state
stays in memory on the Session; this session does not durably persist it.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentique.artifact import Artifact
from agentique.core import Agent, Paused

type SessionState = Literal["running", "paused", "done", "failed"]


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """The serializable projection of a Session held in the shared store."""

    id: str
    agent_name: str
    state: SessionState
    question: str | None = None
    artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class Session:
    """A single agent task in flight, plus its resumable state when paused.

    ``paused`` and ``question`` are set only in the ``paused`` state; ``artifact``
    only once ``done``; ``error`` only when ``failed``. ``kind`` is the artifact
    kind a completed run should produce.
    """

    id: str
    agent: Agent
    kind: str
    state: SessionState
    paused: Paused | None = None
    question: str | None = None
    artifact: Artifact | None = None
    error: str | None = None

    def record(self) -> SessionRecord:
        """Project to the serializable record the store persists."""
        return SessionRecord(
            id=self.id,
            agent_name=self.agent.name,
            state=self.state,
            question=self.question,
            artifact_id=self.artifact.id if self.artifact is not None else None,
        )
