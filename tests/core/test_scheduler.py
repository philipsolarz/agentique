"""The Scheduler: register/dispatch/resume by id, run-lineage, and the single event
tap — exercised offline and deterministically with StubModel."""

import pytest

from agentique.core import (
    Agent,
    Blocked,
    Completed,
    NeedsHuman,
    Scheduler,
)
from agentique.testing import CollectingSink, StubModel
from agentique.tools import AskHuman


def _agent(name: str, *responses, tools=()) -> Agent:
    return Agent(
        name=name,
        instructions="x",
        model=StubModel(list(responses)),
        tools=tools,
    )


async def test_dispatch_runs_a_registered_agent_and_records_the_run() -> None:
    sched = Scheduler()
    sched.register("a", _agent("a", StubModel.text("hi")))
    result = await sched.dispatch("a", "go")
    assert isinstance(result, Completed)
    assert result.output == "hi"
    run = next(iter(sched.runs()))
    assert run.state == "done"
    assert run.agent_id == "a"
    assert run.parent_id is None


async def test_dispatch_unknown_agent_raises() -> None:
    with pytest.raises(KeyError):
        await Scheduler().dispatch("nope", "go")


async def test_pause_then_resume_by_id_completes() -> None:
    sched = Scheduler()
    sched.register(
        "a",
        _agent(
            "a",
            StubModel.tool_call("h1", "ask_human", {"question": "ok?"}),
            StubModel.text("done"),
            tools=(AskHuman(),),
        ),
    )
    paused = await sched.dispatch("a", "go")
    assert isinstance(paused, NeedsHuman)
    run = next(iter(sched.runs()))
    assert run.state == "paused"

    resumed = await sched.resume(run.id, "yes")
    assert isinstance(resumed, Completed)
    assert resumed.output == "done"
    updated = sched.run(run.id)
    assert updated is not None and updated.state == "done"


async def test_resume_unknown_or_not_paused_raises() -> None:
    sched = Scheduler()
    with pytest.raises(KeyError):
        await sched.resume("r99", "x")

    sched.register("a", _agent("a", StubModel.text("done")))
    await sched.dispatch("a", "go")  # completes, never pauses
    run = next(iter(sched.runs()))
    with pytest.raises(ValueError):
        await sched.resume(run.id, "x")


async def test_blocked_run_is_recorded_blocked() -> None:
    sched = Scheduler()
    # an unknown stop reason surfaces as Blocked
    sched.register("a", _agent("a", StubModel.text("", kind="other", raw="weird")))
    result = await sched.dispatch("a", "go")
    assert isinstance(result, Blocked)
    run = next(iter(sched.runs()))
    assert run.state == "blocked"


async def test_events_flow_to_the_scheduler_sink() -> None:
    sink = CollectingSink()
    sched = Scheduler(sink=sink)
    sched.register("a", _agent("a", StubModel.text("hi")))
    await sched.dispatch("a", "go")
    kinds = [type(e).__name__ for e in sink.events]
    # the dispatch event and the tracing events share one tap
    assert "Dispatch" in kinds
    assert "ModelCallStarted" in kinds
    assert "ModelCallFinished" in kinds
