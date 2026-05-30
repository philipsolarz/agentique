"""Artifact: a durable unit of work the harness stores and tracks.

An Artifact is what a run *produces or advances* — a plan, a summary, any payload
an application gives meaning to. It is deliberately **not** a ``Result`` subtype:
``Result`` (in :mod:`agentique.core`) is the ephemeral per-run outcome, whereas an
Artifact persists in the shared store and carries a status that an operator drives
through a lifecycle via the application layer.

``status`` is an application-defined string. ``"proposed"`` is the starting value
and ``"approved"`` / ``"rejected"`` are the common transitions (with the
convenience methods below), but an application may use a richer set — e.g.
``drafting -> review -> approved -> executing -> done`` — by promoting through its
own status strings with :meth:`Artifact.with_status` / ``Coordinator.promote_artifact``.
The harness stays generic: it carries the status and persists it, but assigns no
meaning to the values, just as it assigns none to ``kind`` or ``payload``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# An application-defined lifecycle status. Not constrained to a fixed set: the
# harness moves an artifact between whatever statuses the application defines.
type ArtifactStatus = str


@dataclass(frozen=True, slots=True)
class Artifact:
    """A durable, application-meaningful unit of work with a status lifecycle."""

    id: str
    kind: str
    payload: str
    status: ArtifactStatus = "proposed"

    def with_status(self, status: ArtifactStatus) -> Artifact:
        """Return a copy at ``status`` (the original is unchanged).

        The general promotion primitive; ``approved``/``rejected`` are the common
        cases expressed in terms of it.
        """
        return replace(self, status=status)

    def approved(self) -> Artifact:
        """Return a copy marked ``approved`` (the original is unchanged)."""
        return self.with_status("approved")

    def rejected(self) -> Artifact:
        """Return a copy marked ``rejected`` (the original is unchanged)."""
        return self.with_status("rejected")
