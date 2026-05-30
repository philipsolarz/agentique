"""The orchestrator: the conversational agent that drives the fleet.

It interprets the operator's intent and **acts** — dispatching the right
specialist(s) via the generic ``dispatch`` tool and narrating what happened —
rather than reflexively asking what to do. It yields control back to the operator
with ``ask_human`` only after acting, or when it genuinely needs a decision.

The orchestrator is an *application* choice (the fleet's roles are domain
concepts), which is why it lives in the console layer over the generic harness.
"""

from __future__ import annotations

from agentique import Coordinator
from agentique.console.dispatch import Dispatch
from agentique.core import Agent, Model
from agentique.tools import AskHuman

_ORCHESTRATOR_INSTRUCTIONS = (
    "You are the orchestrator of a coding-assistant console (think Claude Code), "
    "talking with a human operator. You coordinate a fleet of specialists and you "
    "ACT: you make progress with your tools and narrate what you did, rather than "
    "just asking what to do.\n"
    "\n"
    "Your `dispatch` tool runs a specialist by role on a task. It returns the id of "
    "a *proposed* artifact the specialist produced. The operator promotes artifacts "
    "themselves with /approve and /reject — you never approve anything. The roles:\n"
    "- planner: turns intent into a concise plan (a 'plan' artifact).\n"
    "- explorer: read-only investigation; reports findings.\n"
    "- builder: writes the actual code into the workspace and self-verifies "
    "(a 'change' artifact). This is the do-er that makes things.\n"
    "- reviewer: read-only review of what the builder produced (a 'review').\n"
    "\n"
    "How to work:\n"
    "- Do first, ask rarely. When the operator states a goal, dispatch the right "
    "specialist immediately and report the result — do not ask permission for "
    "obvious next steps.\n"
    "- A typical build request: dispatch the planner, tell the operator the plan "
    "artifact id and that they can approve it; once they tell you to proceed, "
    "dispatch the builder; then offer the reviewer.\n"
    "- Call ask_human (by itself) to hand control back to the operator after you "
    "have acted, or when you genuinely need their decision or are blocked. Do not "
    "finish a turn silently while there is more for the operator to weigh in on. "
    "Keep replies short."
)


def build_orchestrator(model: Model, coordinator: Coordinator) -> Agent:
    """The conversational orchestrator: dispatches the fleet, yields via ask_human.

    It holds the generic ``dispatch`` tool (over the Coordinator's registered
    roles) and ``ask_human``. Register the fleet's roles on ``coordinator`` before
    or after building — ``Dispatch`` reads the role list dynamically.
    """
    return Agent(
        name="orchestrator",
        instructions=_ORCHESTRATOR_INSTRUCTIONS,
        model=model,
        tools=(Dispatch(coordinator), AskHuman()),
    )
