"""Agentique harness: the generic, domain-agnostic coordination substrate.

This is the *harness* layer that sits between the framework (:mod:`agentique.core`)
and an application (:mod:`agentique.console`). It coordinates agent runs —
launching them, pausing and resuming them by id, and converging their output in
durable state — on top of the core's seams.

**This layer is deliberately generic and contains no domain specifics.** The name
``code`` signals the coding-agent ecosystem this framework is aimed at; it is *not*
a license to put domain logic here. There are **no repositories, git, diffs, or
pull requests** in this package and there must never be. The harness coordinates
agents that produce *artifacts*; what those artifacts mean is entirely the
application's concern, expressed in :mod:`agentique.console` (or another app built
on this layer).

The dependency arrow points one way: ``console -> code -> core``. This package may
import :mod:`agentique.core`; it must not import :mod:`agentique.console`. The
``Coordinator``, ``Session``, ``Artifact``, and shared ``Store`` live here.
"""

from agentique.code.artifact import Artifact, ArtifactStatus
from agentique.code.coordinator import Coordinator
from agentique.code.session import Session, SessionRecord, SessionState
from agentique.code.store import Store

__all__ = [
    "Artifact",
    "ArtifactStatus",
    "Coordinator",
    "Session",
    "SessionRecord",
    "SessionState",
    "Store",
]
