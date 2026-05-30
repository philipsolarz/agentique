"""Agentique console: the application layer — the only layer that talks to a human.

This is where a conversational ``Console`` and its CLI REPL live. It drives the
harness (:mod:`agentique`) — sending an operator's intent to an orchestrator
agent, surfacing pending decisions, and approving or rejecting the artifacts that
specialist runs produce.

The dependency arrow points one way: ``console -> code -> core``. This package may
import :mod:`agentique` and :mod:`agentique.core`; nothing imports *it*. The
``Console`` and the CLI REPL live here.
"""

from agentique.console.agents import build_orchestrator
from agentique.console.console import Console, Turn, build_console
from agentique.console.dispatch import Dispatch
from agentique.console.fleet import build_fleet

__all__ = [
    "Console",
    "Dispatch",
    "Turn",
    "build_console",
    "build_fleet",
    "build_orchestrator",
]
