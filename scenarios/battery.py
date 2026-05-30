"""Part 3 — the scenario battery.

Six real runs through the same harness, breadth as coverage: a plain completion,
a single tool call, multiple tool calls, tool-error recovery, delegation, and a
bounded loop that hits max_turns. Each produces its own captured run + digest; the
anomalies are then collected into one cross-scenario summary.

    uv run --extra anthropic python -m scenarios.battery
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from agentique.anthropic import AnthropicModel
from agentique.core import Agent, Runtime
from agentique.tools import Delegate, ReadFile
from observability import (
    InMemoryRecorder,
    RecordingModel,
    RecordingTool,
    standing_notes,
)
from scenarios.driver import ScenarioRun, run_scenario
from scenarios.env import load_env, require_env

_READER = (
    "You are a careful assistant. When the user asks about a file, use the "
    "read_file tool to read it before answering."
)


def _model() -> AnthropicModel:
    return AnthropicModel(model=require_env("ANTHROPIC_MODEL"))


async def _text_only(stamp: str) -> ScenarioRun:
    agent = Agent(
        name="explainer",
        instructions="You are a concise technical explainer.",
        model=_model(),
        tools=(),
    )
    return await run_scenario(
        "text_only",
        agent,
        "In two sentences, explain what an idempotency key is and why an HTTP "
        "API uses one.",
        timestamp=stamp,
    )


async def _read_summarize(stamp: str) -> ScenarioRun:
    agent = Agent(
        name="reader", instructions=_READER, model=_model(), tools=(ReadFile(),)
    )
    return await run_scenario(
        "read_summarize",
        agent,
        "Read pyproject.toml and tell me the project name, version, and whether "
        "it has any runtime dependencies.",
        timestamp=stamp,
    )


async def _multi_tool(stamp: str) -> ScenarioRun:
    agent = Agent(
        name="reader", instructions=_READER, model=_model(), tools=(ReadFile(),)
    )
    return await run_scenario(
        "multi_tool",
        agent,
        "Read both README.md and pyproject.toml, then in 3 bullets describe what "
        "this project is and how it is built.",
        timestamp=stamp,
    )


async def _tool_error_recovery(stamp: str) -> ScenarioRun:
    agent = Agent(
        name="reader", instructions=_READER, model=_model(), tools=(ReadFile(),)
    )
    return await run_scenario(
        "tool_error_recovery",
        agent,
        "Read the file CHANGELOG.md and summarize it. If that file does not "
        "exist, read README.md instead and summarize that.",
        timestamp=stamp,
    )


async def _delegate(stamp: str) -> ScenarioRun:
    # Share one recorder across parent and child so the child's turns interleave
    # into the same capture (per DP3). The child's model/tools are wrapped here,
    # before the Delegate is built; run_scenario wraps the parent with the same one.
    recorder = InMemoryRecorder()
    child = Agent(
        name="summarizer",
        instructions=_READER,
        model=RecordingModel(_model(), recorder),
        tools=(RecordingTool(ReadFile(), recorder),),
    )
    delegate = Delegate(
        child,
        name="summarizer",
        description="Summarize a file. Pass the task (which file, how) as 'prompt'.",
    )
    parent = Agent(
        name="coordinator",
        instructions="Delegate file-summary tasks to the 'summarizer' tool.",
        model=_model(),
        tools=(delegate,),
    )
    return await run_scenario(
        "delegate",
        parent,
        "Use the summarizer to summarize README.md in 3 bullets.",
        timestamp=stamp,
        recorder=recorder,
    )


async def _max_turns_blocked(stamp: str) -> ScenarioRun:
    agent = Agent(
        name="reader", instructions=_READER, model=_model(), tools=(ReadFile(),)
    )
    return await run_scenario(
        "max_turns_blocked",
        agent,
        "Read README.md and summarize it.",
        runtime=Runtime(max_turns=1),
        timestamp=stamp,
    )


def _write_summary(runs_root: Path, stamp: str, runs: list[ScenarioRun]) -> Path:
    lines = [
        f"# Battery summary — {stamp}",
        "",
        "| scenario | outcome | turns | tool calls | anomalies |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        r = run.record
        lines.append(
            f"| {r.scenario} | {r.outcome} | {r.turns} | {r.tool_calls} "
            f"| {len(r.anomalies)} |"
        )
    lines += ["", "## Per-scenario anomalies"]
    for run in runs:
        lines.append(f"### {run.record.scenario}")
        if run.record.anomalies:
            lines.extend(f"- {a}" for a in run.record.anomalies)
        else:
            lines.append("_none_")
    lines += ["", "## Standing notes (framework-wide)"]
    lines.extend(f"- {note}" for note in standing_notes())
    lines += [
        "",
        "## Observations",
        "- delegation: the trace interleaves parent and child turns in one flat "
        "event stream; the capture format has no notion of agent nesting, and the "
        "delegate tool call is recorded after the child's turns complete.",
        "",
    ]
    summary_path = runs_root / f"{stamp}-summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


async def main() -> None:
    load_env()
    require_env("ANTHROPIC_API_KEY")
    require_env("ANTHROPIC_MODEL")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs = [
        await _text_only(stamp),
        await _read_summarize(stamp),
        await _multi_tool(stamp),
        await _tool_error_recovery(stamp),
        await _delegate(stamp),
        await _max_turns_blocked(stamp),
    ]
    for run in runs:
        print(f"{run.record.scenario:>20}  {run.record.outcome:<10}  {run.out_dir}")

    summary_path = _write_summary(runs[0].out_dir.parent, stamp, runs)
    print()
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
