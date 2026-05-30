"""Coordinator: dispatch lands a proposed artifact, pause/resume-by-id continues a
specific run, bad resumes raise, blocked runs fail, and artifacts approve/reject.
Exercised offline with StubModel / ask_human.
"""

import pytest

from agentique import Coordinator
from agentique.core import Agent, ModelResponse
from agentique.testing import StubModel
from agentique.tools import AskHuman


def _planner(*responses: ModelResponse) -> Agent:
    return Agent(
        name="planner",
        instructions="plan",
        model=StubModel(list(responses)),
        tools=(AskHuman(),),
    )


async def test_dispatch_completes_into_a_proposed_artifact() -> None:
    coord = Coordinator()
    session = await coord.dispatch(
        _planner(StubModel.text("the plan")), "make a plan", kind="plan"
    )

    assert session.state == "done"
    assert session.artifact is not None
    assert session.artifact.kind == "plan"
    assert session.artifact.payload == "the plan"
    assert session.artifact.status == "proposed"
    # it converged in the shared store, not just on the session.
    assert await coord.store.get_artifact(session.artifact.id) == session.artifact


async def test_dispatch_pauses_then_resume_by_id_completes() -> None:
    coord = Coordinator()
    agent = _planner(
        StubModel.tool_call("h1", "ask_human", {"question": "which file?"}),
        StubModel.text("planned around notes.txt"),
    )
    paused = await coord.dispatch(agent, "plan it", kind="plan")
    assert paused.state == "paused"
    assert paused.question == "which file?"
    assert paused.artifact is None

    done = await coord.resume(paused.id, "notes.txt")
    assert done.id == paused.id  # the same session, resumed by id
    assert done.state == "done"
    assert done.artifact is not None
    assert done.artifact.payload == "planned around notes.txt"


async def test_resume_unknown_session_raises() -> None:
    with pytest.raises(KeyError):
        await Coordinator().resume("nope", "x")


async def test_resume_non_paused_session_raises() -> None:
    coord = Coordinator()
    session = await coord.dispatch(_planner(StubModel.text("done")), "go")
    assert session.state == "done"
    with pytest.raises(ValueError, match="not paused"):
        await coord.resume(session.id, "x")


async def test_blocked_run_marks_session_failed() -> None:
    coord = Coordinator()
    # the model calls a tool the agent doesn't hold -> Blocked(unknown tool).
    agent = Agent(
        name="x",
        instructions="",
        model=StubModel([StubModel.tool_call("c1", "ghost", {})]),
    )
    session = await coord.dispatch(agent, "go")
    assert session.state == "failed"
    assert session.error is not None
    assert "unknown tool" in session.error


async def test_approve_and_reject_artifact_persist() -> None:
    coord = Coordinator()
    session = await coord.dispatch(_planner(StubModel.text("p")), "go", kind="plan")
    assert session.artifact is not None
    artifact_id = session.artifact.id

    approved = await coord.approve_artifact(artifact_id)
    assert approved.status == "approved"
    stored = await coord.store.get_artifact(artifact_id)
    assert stored is not None and stored.status == "approved"

    rejected = await coord.reject_artifact(artifact_id)
    assert rejected.status == "rejected"


async def test_promote_artifact_to_application_defined_status() -> None:
    coord = Coordinator()
    session = await coord.dispatch(_planner(StubModel.text("p")), "go", kind="change")
    assert session.artifact is not None

    promoted = await coord.promote_artifact(session.artifact.id, "executing")
    assert promoted.status == "executing"
    stored = await coord.store.get_artifact(session.artifact.id)
    assert stored is not None and stored.status == "executing"


async def test_promote_unknown_artifact_raises() -> None:
    with pytest.raises(KeyError, match="no artifact"):
        await Coordinator().promote_artifact("nope", "approved")


async def test_two_sessions_get_distinct_ids() -> None:
    coord = Coordinator()
    a = await coord.dispatch(_planner(StubModel.text("a")), "a")
    b = await coord.dispatch(_planner(StubModel.text("b")), "b")
    assert a.id != b.id
    assert {s.id for s in coord.sessions()} == {a.id, b.id}
