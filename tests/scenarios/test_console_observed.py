"""The observed console writes a per-session trace capturing BOTH agents' seam
crossings, and an ask_human pause is not flagged as a (stale) anomaly. Offline.
"""

import json
from pathlib import Path

from agentique.testing import StubModel
from scenarios.console import wire, write_session_trace


def _planner_model(path: str) -> StubModel:
    return StubModel(
        [
            StubModel.tool_call("r1", "read_file", {"path": path}),
            StubModel.text("Plan:\n1. Do A"),
        ]
    )


def _orchestrator_model(path: str) -> StubModel:
    return StubModel(
        [
            StubModel.tool_call("p1", "plan_file", {"path": path}),
            StubModel.tool_call("h1", "ask_human", {"question": "Approve?"}),
        ]
    )


async def test_observed_console_writes_multi_agent_trace(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("notes", encoding="utf-8")
    console, recorder = wire(
        _orchestrator_model(str(notes)), planner_model=_planner_model(str(notes))
    )

    turn = await console.send(f"plan {notes}")
    assert turn.awaiting_input is True
    assert console.last_result is not None

    out = write_session_trace(tmp_path / "trace", recorder, console.last_result, 0.5)
    assert (out / "events.jsonl").exists()
    assert (out / "manifest.json").exists()
    assert (out / "digest.md").exists()

    events = [
        json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()
    ]
    tool_names = {e["tool_name"] for e in events if e["kind"] == "tool_call"}
    # the orchestrator's tools AND the nested planner's tool are all captured.
    assert {"plan_file", "ask_human", "read_file"} <= tool_names

    manifest = json.loads((out / "manifest.json").read_text())
    # the ask_human pause must not surface as the old "unrecoverable" anomaly.
    assert not any("unrecoverable" in a for a in manifest["anomalies"])
    assert not any("PauseRequested" in a for a in manifest["anomalies"])


async def test_trace_digest_mentions_the_planner_subagent(tmp_path: Path) -> None:
    notes = tmp_path / "n.txt"
    notes.write_text("x", encoding="utf-8")
    console, recorder = wire(
        _orchestrator_model(str(notes)), planner_model=_planner_model(str(notes))
    )
    await console.send("plan it")
    assert console.last_result is not None

    out = write_session_trace(tmp_path / "t", recorder, console.last_result, 0.1)
    digest = (out / "digest.md").read_text()
    assert "read_file" in digest
    assert "plan_file" in digest
