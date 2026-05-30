"""The digest: a compact, anomaly-forward Markdown view of one run, written for a
Claude Code session to read on its own — the full ``events.jsonl`` is on disk if a
deeper look is needed.

Layout (kept small on purpose):

    # <scenario> — <outcome>
    turns N · tool calls T · <wall>s

    ## Trace
    turn 1: model→tool_use 0.8s | read_file(path=foo.txt)→ok 1.2kb 40ms
    turn 2: model→end_turn 0.6s (2 blocks, 412 chars)

    ## Anomalies & contract deltas
    - <finding>

Target budget: a one-line header pair, one line per turn, then the findings —
typically a few hundred tokens for a 2-6 turn run.
"""

from __future__ import annotations

from collections.abc import Sequence

from observability.events import (
    CallEvent,
    ModelCallEvent,
    RunRecord,
    Turn,
    group_into_turns,
)


def _fmt_dur(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


def _fmt_size(chars: int) -> str:
    return f"{chars}b" if chars < 1024 else f"{chars / 1024:.1f}kb"


def _fmt_args(arguments: dict[str, object]) -> str:
    parts = []
    for key, value in list(arguments.items())[:3]:
        text = str(value)
        if len(text) > 40:
            text = text[:37] + "..."
        parts.append(f"{key}={text}")
    rendered = " ".join(parts)
    if len(arguments) > 3:
        rendered += " …"
    return rendered


def _model_detail(model: ModelCallEvent) -> str:
    if model.raised is not None:
        return ""
    if model.stop_reason == "tool_use":
        return ""
    total = sum(b.size for b in model.blocks)
    return f"({len(model.blocks)} block(s), {total} chars)"


def _turn_line(number: int, turn: Turn) -> str:
    model = turn.model
    if model.raised is not None:
        head = f"turn {number}: model→RAISED {model.raised} {_fmt_dur(model.latency_s)}"
    else:
        detail = _model_detail(model)
        head = (
            f"turn {number}: model→{model.stop_reason} {_fmt_dur(model.latency_s)}"
            + (f" {detail}" if detail else "")
        )
    segments = [head]
    for tool in turn.tools:
        if tool.raised is not None:
            outcome = f"RAISED {tool.raised}"
            size = ""
        else:
            outcome = "ERR" if tool.is_error else "ok"
            size = f" {_fmt_size(tool.result_len or 0)}"
        segments.append(
            f"{tool.tool_name}({_fmt_args(dict(tool.arguments))})→{outcome}"
            f"{size} {_fmt_dur(tool.latency_s)}"
        )
    return " | ".join(segments)


def render_digest(record: RunRecord, events: Sequence[CallEvent]) -> str:
    """Render the full ``digest.md`` text for a run."""
    lines = [
        f"# {record.scenario} — {record.outcome}",
        f"turns {record.turns} · tool calls {record.tool_calls} "
        f"· {record.wall_time_s:.1f}s",
        "",
        "## Trace",
    ]
    turns = group_into_turns(events)
    if turns:
        lines.extend(_turn_line(i, turn) for i, turn in enumerate(turns, start=1))
    else:
        lines.append("_(no model calls recorded)_")
    lines += ["", "## Anomalies & contract deltas"]
    if record.anomalies:
        lines.extend(f"- {item}" for item in record.anomalies)
    else:
        lines.append("_none_")
    return "\n".join(lines) + "\n"
