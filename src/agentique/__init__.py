"""Agentique: an agent *harness* framework built on top of ``agentique.core``.

``agentique.core`` is the application-agnostic agent framework — the seam Protocols
(Model, Tool, Memory), the value types, and the Runtime that drives a single agent.
This package, the ``agentique`` root, is the generic harness that *coordinates*
agents built on that framework: launching them, pausing and resuming them, and
converging their output into durable state. Its public surface lives right here —
``from agentique import Coordinator, Session, Artifact, Store, Role``.

The harness is deliberately domain-agnostic: it coordinates agents that produce
*artifacts*; what an artifact *means* is the application's concern, expressed in
:mod:`agentique.console` (or another app built on this layer). The dependency arrow
runs ``console -> agentique (harness) -> agentique.core``; the harness may import
:mod:`agentique.core` but never :mod:`agentique.console`.

Satellites of the core framework live in their own submodules:
:mod:`agentique.tools`, :mod:`agentique.memory`, :mod:`agentique.testing`, and the
optional :mod:`agentique.anthropic` provider (behind the ``anthropic`` extra).
"""

from agentique.artifact import Artifact, ArtifactStatus, TextPayload, payload_text
from agentique.coordinator import Coordinator
from agentique.role import Role
from agentique.session import PausedRun, Session, SessionRecord, SessionState
from agentique.store import KindRegistry, Store

__all__ = [
    "Artifact",
    "ArtifactStatus",
    "Coordinator",
    "KindRegistry",
    "PausedRun",
    "Role",
    "Session",
    "SessionRecord",
    "SessionState",
    "Store",
    "TextPayload",
    "payload_text",
]
