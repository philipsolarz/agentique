"""Artifact: a durable unit of work the harness stores and tracks.

An Artifact is what a run *produces or advances* — a plan, a summary, any payload
an application gives meaning to. It is deliberately **not** a ``Result`` subtype:
``Result`` (in :mod:`agentique.core`) is the ephemeral per-run outcome, whereas an
Artifact persists in the shared store and carries a small status lifecycle,
``proposed -> approved | rejected``, that a human drives via the application layer.

The harness is generic: what an artifact's ``kind`` and ``payload`` *mean* is the
application's concern, never this layer's.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

type ArtifactStatus = Literal["proposed", "approved", "rejected"]


@dataclass(frozen=True, slots=True)
class Artifact:
    """A durable, application-meaningful unit of work with a status lifecycle."""

    id: str
    kind: str
    payload: str
    status: ArtifactStatus = "proposed"

    def approved(self) -> Artifact:
        """Return a copy marked ``approved`` (the original is unchanged)."""
        return replace(self, status="approved")

    def rejected(self) -> Artifact:
        """Return a copy marked ``rejected`` (the original is unchanged)."""
        return replace(self, status="rejected")
