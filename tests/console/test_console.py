"""Console: the orchestrator dispatches fleet specialists via the generic dispatch
tool, lands their output as proposed artifacts, pauses via ask_human, threads the
conversation across resumes, and approves/rejects artifacts. The Builder actually
writes into the workspace. Offline via StubModel.
"""

from pathlib import Path

import pytest

from agentique import payload_text
from agentique.console.console import Console, build_console
from agentique.testing import StubModel


def _console(
    orch: StubModel, fleet: StubModel, *, workspace_root: str = "workspace"
) -> Console:
    return build_console(orch, fleet_model=fleet, workspace_root=workspace_root)


async def test_orchestrator_dispatches_planner_then_pauses_for_approval() -> None:
    console = _console(
        StubModel(
            [
                StubModel.tool_call(
                    "d1", "dispatch", {"role": "planner", "task": "build a snake game"}
                ),
                StubModel.tool_call(
                    "h1", "ask_human", {"question": "Drafted a plan. Approve it?"}
                ),
            ]
        ),
        StubModel([StubModel.text("Plan:\n1. Make the board\n2. Move the snake")]),
    )

    turn = await console.send("build me a snake game")
    assert turn.awaiting_input is True
    assert turn.done is False
    assert "Approve it?" in turn.message

    artifacts = await console.artifacts()
    assert len(artifacts) == 1
    assert artifacts[0].kind == "plan"
    assert artifacts[0].status == "proposed"
    assert "Make the board" in payload_text(artifacts[0].payload)

    approved = await console.approve(artifacts[0].id)
    assert approved.status == "approved"


async def test_builder_writes_into_the_workspace(tmp_path: Path) -> None:
    console = _console(
        StubModel(
            [
                StubModel.tool_call(
                    "d1",
                    "dispatch",
                    {"role": "builder", "task": "write snake_game.html"},
                ),
                StubModel.tool_call(
                    "h1", "ask_human", {"question": "Built it. Approve?"}
                ),
            ]
        ),
        StubModel(
            [
                StubModel.tool_call(
                    "w1",
                    "write_file",
                    {"path": "snake_game.html", "content": "<html>snake</html>"},
                ),
                StubModel.text("Built snake_game.html in the workspace."),
            ]
        ),
        workspace_root=str(tmp_path),
    )

    turn = await console.send("build me a snake game in HTML")
    assert turn.awaiting_input is True
    # the file really landed on disk, in the confined workspace.
    assert (tmp_path / "snake_game.html").read_text(encoding="utf-8") == (
        "<html>snake</html>"
    )
    artifacts = await console.artifacts()
    assert len(artifacts) == 1
    assert artifacts[0].kind == "change"
    assert "Built snake_game.html" in payload_text(artifacts[0].payload)


async def test_unknown_role_is_surfaced_not_dispatched() -> None:
    console = _console(
        StubModel(
            [
                StubModel.tool_call("d1", "dispatch", {"role": "ghost", "task": "x"}),
                StubModel.tool_call("h1", "ask_human", {"question": "what now?"}),
            ]
        ),
        StubModel([StubModel.text("unused")]),
    )
    turn = await console.send("do something")
    assert turn.awaiting_input is True
    # no artifact produced; the orchestrator was told the role is unknown.
    assert await console.artifacts() == ()


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


async def test_reject_artifact_via_console() -> None:
    console = _console(
        StubModel(
            [
                StubModel.tool_call(
                    "d1", "dispatch", {"role": "planner", "task": "go"}
                ),
                StubModel.tool_call("h1", "ask_human", {"question": "ok?"}),
            ]
        ),
        StubModel([StubModel.text("Plan: 1. do it")]),
    )
    await console.send("plan it")
    artifacts = await console.artifacts()
    rejected = await console.reject(artifacts[0].id)
    assert rejected.status == "rejected"
