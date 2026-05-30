"""Post-run output: turn a finished run's recorded events into the three on-disk
artifacts. All file I/O lives here and runs **once, after the run completes** —
never on the recorder's timed path.

``events.jsonl`` is full fidelity (one event per line, totally serialized),
``manifest.json`` is the :class:`RunRecord` rollup, and ``digest.md`` is the
compact anomaly-forward view for a Claude Code session.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from agentique.core.result import Result
from observability.anomalies import detect_anomalies
from observability.digest import render_digest
from observability.events import (
    CallEvent,
    ModelCallEvent,
    RunRecord,
    ToolCallEvent,
    run_record_to_dict,
    to_jsonl_line,
)


def build_run_record(
    scenario: str,
    events: Sequence[CallEvent],
    result: Result,
    wall_time_s: float,
) -> RunRecord:
    """Roll up a run's events and terminal Result into a :class:`RunRecord`."""
    return RunRecord(
        scenario=scenario,
        outcome=type(result).__name__,
        turns=sum(isinstance(e, ModelCallEvent) for e in events),
        tool_calls=sum(isinstance(e, ToolCallEvent) for e in events),
        wall_time_s=wall_time_s,
        anomalies=detect_anomalies(events, result),
    )


def write_run(out_dir: Path, record: RunRecord, events: Sequence[CallEvent]) -> Path:
    """Write ``events.jsonl``, ``manifest.json`` and ``digest.md`` into ``out_dir``
    (created if absent). Returns ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl = "".join(to_jsonl_line(event) + "\n" for event in events)
    (out_dir / "events.jsonl").write_text(jsonl, encoding="utf-8")

    manifest = json.dumps(run_record_to_dict(record), indent=2, default=repr)
    (out_dir / "manifest.json").write_text(manifest + "\n", encoding="utf-8")

    (out_dir / "digest.md").write_text(render_digest(record, events), encoding="utf-8")
    return out_dir
