"""Durable pause/resume: an approval gate survives a process restart.

A run pauses in one Coordinator; a *fresh* Coordinator + Scheduler over the same
backing Memory loads the persisted paused run, the app re-registers the agent under
the same id, and the run resumes to completion — proving the Paused snapshot (not
the live Agent) is what persists. Deterministic via StubModel: the re-registered
agent only needs the post-resume continuation, since the pre-pause turns are baked
into the persisted snapshot and never replayed.
"""

from agentique import Coordinator, Role, Store
from agentique.core import Agent, Scheduler
from agentique.memory import InMemoryStore
from agentique.testing import StubModel


def _pausing_agent() -> Agent:
    # one turn: ask the human, then pause.
    return Agent(
        name="planner",
        instructions="plan",
        model=StubModel(
            [StubModel.tool_call("h1", "ask_human", {"question": "which file?"})]
        ),
        tools=(_ask_human(),),
    )


def _resuming_agent() -> Agent:
    # the fresh-process agent: it only produces the continuation after the answer.
    return Agent(
        name="planner",
        instructions="plan",
        model=StubModel([StubModel.text("planned around notes.txt")]),
        tools=(_ask_human(),),
    )


def _ask_human():
    from agentique.tools import AskHuman

    return AskHuman()


async def test_paused_run_survives_a_fresh_coordinator() -> None:
    memory = InMemoryStore()  # the shared, durable backing store

    # --- process 1: dispatch to a pause; the snapshot is persisted ---
    coord1 = Coordinator(store=Store(memory), scheduler=Scheduler())
    coord1.register_role(Role(name="planner", agent=_pausing_agent(), kind="plan"))
    paused = await coord1.dispatch_role("planner", "make a plan")
    assert paused.state == "paused"
    run_id = paused.id
    # it is durably recorded, not just in memory.
    assert await Store(memory).get_paused_run(run_id) is not None

    # --- process 2: a brand-new Coordinator + Scheduler over the same Memory ---
    coord2 = Coordinator(store=Store(memory), scheduler=Scheduler())
    # the app re-registers its agents under the same ids before loading.
    coord2.register_role(Role(name="planner", agent=_resuming_agent(), kind="plan"))
    loaded = await coord2.load_paused_runs()
    assert [s.id for s in loaded] == [run_id]
    assert coord2.session(run_id) is not None

    # resume by id in the fresh process completes the run and lands the artifact.
    done = await coord2.resume(run_id, "notes.txt")
    assert done.state == "done"
    assert done.artifact is not None
    assert done.artifact.kind == "plan"
    # and the persisted paused run is cleared once it resolves.
    assert await Store(memory).get_paused_run(run_id) is None
