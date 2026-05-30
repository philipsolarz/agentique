"""Writer + digest: the three artifacts are produced once after a run, the
manifest round-trips, and the digest renders header/trace/anomalies.
"""

import json

from agentique.core.context import Context
from agentique.core.result import Completed
from observability.digest import render_digest
from observability.events import ContentBlockSummary, ModelCallEvent, ToolCallEvent
from observability.writer import build_run_record, write_run

_EVENTS = [
    ModelCallEvent(
        system_len=20,
        message_count=1,
        tool_names=("read_file",),
        stop_reason="tool_use",
        blocks=(ContentBlockSummary("tool_use", 18),),
        latency_s=0.8,
    ),
    ToolCallEvent(
        tool_name="read_file",
        arguments={"path": "notes.txt"},
        result_len=1230,
        is_error=False,
        latency_s=0.04,
    ),
    ModelCallEvent(
        system_len=20,
        message_count=3,
        tool_names=("read_file",),
        stop_reason="end_turn",
        blocks=(ContentBlockSummary("text", 412),),
        latency_s=0.6,
    ),
]


def test_build_run_record_counts_and_outcome() -> None:
    record = build_run_record("demo", _EVENTS, Completed("done", Context()), 1.5)
    assert record.scenario == "demo"
    assert record.outcome == "Completed"
    assert record.turns == 2
    assert record.tool_calls == 1
    assert record.wall_time_s == 1.5


def test_write_run_emits_three_files_and_manifest_round_trips(tmp_path) -> None:
    record = build_run_record("demo", _EVENTS, Completed("done", Context()), 1.5)
    out = write_run(tmp_path / "run", record, _EVENTS)

    assert (out / "events.jsonl").exists()
    assert (out / "manifest.json").exists()
    assert (out / "digest.md").exists()

    # One JSONL line per event.
    lines = (out / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["outcome"] == "Completed"
    assert manifest["turns"] == 2
    assert manifest["tool_calls"] == 1


def test_digest_renders_header_trace_and_anomalies() -> None:
    record = build_run_record("demo", _EVENTS, Completed("done", Context()), 1.5)
    text = render_digest(record, _EVENTS)

    assert text.startswith("# demo — Completed")
    assert "## Trace" in text
    assert "turn 1: model→tool_use" in text
    assert "read_file(path=notes.txt)→ok 1.2kb" in text
    assert "turn 2: model→end_turn" in text
    assert "## Anomalies & contract deltas" in text
