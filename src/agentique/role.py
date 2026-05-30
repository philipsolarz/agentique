"""Role: a named specialist the Coordinator can dispatch.

The fleet of agents an application runs differ only by *configuration* — a role
prompt, the tools they may use, and the kind of artifact they produce. Rather than
a bespoke class per specialist, a ``Role`` bundles a declarative
:class:`~agentique.core.agent.Agent` with the artifact ``kind`` its runs produce,
under a dispatch ``name`` the application (or an orchestrator) refers to it by. The
Coordinator can register roles and dispatch one *by name* — making "run the
builder on this task" an explicit, visible coordination act rather than something
buried inside a tool call.

This layer stays domain-agnostic: a Role is just (name, agent, kind). What the
``kind`` *means* — ``"plan"``, ``"change"``, ``"review"`` — is the application's
concern, defined in :mod:`agentique.console`.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentique.core import Agent


@dataclass(frozen=True, slots=True)
class Role:
    """A dispatchable specialist: an Agent plus the artifact kind it produces."""

    name: str
    agent: Agent
    kind: str = "result"
