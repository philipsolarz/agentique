"""The scenario driver: wrap an agent in the recording layer, drive it through the
real Runtime, and write the run's three artifacts.

It takes a fully-built :class:`Agent` and instruments it by composition — the
agent's model and tools are wrapped with a single shared recorder (so events
interleave in run order) and the rest of the agent is left untouched. The original
agent is never mutated; a wrapped copy is made with :func:`dataclasses.replace`.

A scenario that delegates can wrap its child's model with the *same* recorder
before building the ``Delegate`` tool, to make the child's calls visible too;
otherwise a delegated run shows up as a single tool call here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from agentique.core import Agent, Runtime
from agentique.core.result import Result
from observability import (
    InMemoryRecorder,
    RunRecord,
    build_run_record,
    instrument_agent,
    write_run,
)

# scenarios/ sits at the repo root; runs/ is its sibling (gitignored output).
_RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    """The outcome of a captured scenario: the terminal Result, the directory the
    artifacts were written to, and the rollup."""

    result: Result
    out_dir: Path
    record: RunRecord


async def run_scenario(
    scenario: str,
    agent: Agent,
    prompt: str,
    *,
    runtime: Runtime | None = None,
    runs_root: Path | None = None,
    timestamp: str | None = None,
    recorder: InMemoryRecorder | None = None,
) -> ScenarioRun:
    """Run ``agent`` on ``prompt`` under the recording layer and write the captured
    run to ``<runs_root>/<timestamp>-<scenario>/``. ``timestamp`` defaults to now;
    pass a fixed value for a deterministic directory (tests).

    Pass ``recorder`` to share one across agents — e.g. a delegation scenario that
    wrapped its child's model/tools with the same recorder before building the
    parent, so the child's calls interleave into this capture."""
    runtime = runtime if runtime is not None else Runtime()
    runs_root = runs_root if runs_root is not None else _RUNS_ROOT

    recorder = recorder if recorder is not None else InMemoryRecorder()
    instrumented = instrument_agent(agent, recorder)

    start = time.perf_counter()
    result = await runtime.run(instrumented, prompt)
    wall_time_s = time.perf_counter() - start

    record = build_run_record(
        scenario,
        recorder.events,
        result,
        wall_time_s,
    )
    stamp = timestamp if timestamp is not None else time.strftime("%Y%m%d-%H%M%S")
    out_dir = write_run(runs_root / f"{stamp}-{scenario}", record, recorder.events)
    return ScenarioRun(result=result, out_dir=out_dir, record=record)
