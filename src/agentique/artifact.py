"""Artifact: a durable unit of work the harness stores and tracks.

An Artifact is what a run *produces or advances*. Its ``payload`` is a typed
Pydantic model — the harness stays generic *over* the payload type and never
defines what a ``kind`` means; the concrete models live in the application (e.g.
:mod:`agentique.console`). ``TextPayload`` is the harness's default carrier for a
plain-text result (and the fallback for an unregistered kind), so the common case
needs no app model.

Artifacts form a provenance **DAG**: ``derived_from`` lists the ids of the
artifacts this one was derived from (e.g. a review derived from a change derived
from a plan). The harness records these edges verbatim — they are supplied by the
dispatching app, never inferred — and assigns them no meaning.

``status`` is an application-defined string (``"proposed"`` start; ``"approved"`` /
``"rejected"`` common transitions). The harness carries and persists it, assigning
no meaning to the value, just as it assigns none to ``kind``, ``payload``, or the
provenance edges. ``Artifact`` stays a frozen *dataclass* (not a Pydantic model):
it merely carries a Pydantic ``payload``, so its ``with_status``/``replace``
lifecycle is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pydantic import BaseModel, ConfigDict

# An application-defined lifecycle status. Not constrained to a fixed set.
type ArtifactStatus = str


class TextPayload(BaseModel):
    """The harness's default artifact payload: a plain-text result. Used directly
    for free-form output and as the fallback when a kind has no app model."""

    model_config = ConfigDict(frozen=True)

    text: str


@dataclass(frozen=True, slots=True)
class Artifact:
    """A durable, application-meaningful unit of work with a typed payload, a
    provenance DAG, and a status lifecycle."""

    id: str
    kind: str
    payload: BaseModel
    status: ArtifactStatus = "proposed"
    derived_from: tuple[str, ...] = ()

    def with_status(self, status: ArtifactStatus) -> Artifact:
        """Return a copy at ``status`` (the original is unchanged)."""
        return replace(self, status=status)

    def approved(self) -> Artifact:
        return self.with_status("approved")

    def rejected(self) -> Artifact:
        return self.with_status("rejected")


def payload_text(payload: BaseModel) -> str:
    """A human-readable rendering of an artifact payload, for previews. A
    ``TextPayload`` renders as its text; any other model renders as its JSON."""
    if isinstance(payload, TextPayload):
        return payload.text
    return payload.model_dump_json()
