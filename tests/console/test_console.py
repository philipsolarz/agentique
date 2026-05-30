"""Console: the orchestrator dispatches the planner (which reads a real file) into
a proposed artifact, pauses for the operator via ask_human, threads the
conversation across resumes, and approves/rejects artifacts. Offline via StubModel.
"""

from pathlib import Path

import pytest

from agentique.code import Coordinator
from agentique.console.agents import build_orchestrator, build_planner
from agentique.console.console import Console
from agentique.testing import StubModel


def _planner_model(path: str) -> StubModel:
    return StubModel(
        [
            StubModel.tool_call("r1", "read_file", {"path": path}),
            StubModel.text("Plan:\n1. Do A\n2. Do B"),
        ]
    )


def _console(orch_model: StubModel, planner_model: StubModel) -> Console:
    coordinator = Coordinator()
    orchestrator = build_orchestrator(
        orch_model, coordinator, build_planner(planner_model)
    )
    return Console(orchestrator, coordinator=coordinator)


async def test_orchestrator_plans_then_pauses_for_approval(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("some notes", encoding="utf-8")

    console = _console(
        StubModel(
            [
                StubModel.tool_call("p1", "plan_file", {"path": str(notes)}),
                StubModel.tool_call(
                    "h1", "ask_human", {"question": "Drafted a plan. Approve it?"}
                ),
            ]
        ),
        _planner_model(str(notes)),
    )

    turn = await console.send(f"plan {notes}")
    assert turn.awaiting_input is True
    assert turn.done is False
    assert "Approve it?" in turn.message

    artifacts = await console.artifacts()
    assert len(artifacts) == 1
    assert artifacts[0].kind == "plan"
    assert artifacts[0].status == "proposed"
    assert "Do A" in artifacts[0].payload

    approved = await console.approve(artifacts[0].id)
    assert approved.status == "approved"


async def test_send_resumes_and_threads_conversation() -> None:
    console = _console(
        StubModel(
            [
                StubModel.tool_call(
                    "h1", "ask_human", {"question": "What do you need?"}
                ),
                StubModel.tool_call("h2", "ask_human", {"question": "Anything else?"}),
            ]
        ),
        StubModel([StubModel.text("unused")]),
    )

    first = await console.send("hello")
    assert first.awaiting_input is True
    assert "What do you need?" in first.message

    second = await console.send("just chatting")
    assert second.awaiting_input is True
    assert "Anything else?" in second.message


async def test_send_after_conversation_ends_raises() -> None:
    console = _console(
        StubModel([StubModel.text("goodbye")]),  # ends immediately (Completed)
        StubModel([StubModel.text("unused")]),
    )
    turn = await console.send("hi")
    assert turn.done is True
    assert "goodbye" in turn.message
    with pytest.raises(RuntimeError, match="ended"):
        await console.send("more")


async def test_reject_artifact_via_console(tmp_path: Path) -> None:
    notes = tmp_path / "n.txt"
    notes.write_text("x", encoding="utf-8")
    console = _console(
        StubModel(
            [
                StubModel.tool_call("p1", "plan_file", {"path": str(notes)}),
                StubModel.tool_call("h1", "ask_human", {"question": "ok?"}),
            ]
        ),
        _planner_model(str(notes)),
    )
    await console.send("plan it")
    artifacts = await console.artifacts()
    rejected = await console.reject(artifacts[0].id)
    assert rejected.status == "rejected"
