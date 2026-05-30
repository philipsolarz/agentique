"""Dev-only: the agentique console REPL, observed.

Wraps the *same* ``Console`` (orchestrator + planner) with the observability
recording layer — by composition, so the shipped package imports nothing from here
— and writes a single per-session trace under ``runs/<ts>-console/`` (refreshed
after each turn so it survives Ctrl-C), printing the digest on exit. Run via
``make console``.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace
from pathlib import Path

from agentique import Coordinator
from agentique.console.agents import build_orchestrator
from agentique.console.cli import load_env
from agentique.console.console import Console
from agentique.console.fleet import build_fleet
from agentique.core import Model, Result, Runtime
from agentique.tools import Workspace
from observability import (
    InMemoryRecorder,
    build_run_record,
    instrument_agent,
    render_digest,
    write_run,
)

# scenarios/ sits at the repo root; runs/ is its sibling (gitignored output).
_RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"


def wire(
    model: Model,
    *,
    fleet_model: Model | None = None,
    workspace_root: str = "workspace",
) -> tuple[Console, InMemoryRecorder]:
    """Build the Console with the orchestrator *and* the whole fleet instrumented.

    Wrapping every agent with the same recorder interleaves each dispatched
    specialist's nested model/tool calls into the orchestrator's stream, so one
    session trace shows the whole multi-agent picture. ``fleet_model`` defaults to
    ``model``; pass a separate one to script the specialists apart in tests.
    """
    recorder = InMemoryRecorder()
    workspace = Workspace(workspace_root)
    coordinator = Coordinator(runtime=Runtime(max_turns=24))
    for role in build_fleet(
        fleet_model if fleet_model is not None else model, workspace
    ):
        coordinator.register_role(
            replace(role, agent=instrument_agent(role.agent, recorder))
        )
    orchestrator = instrument_agent(build_orchestrator(model, coordinator), recorder)
    return (
        Console(orchestrator, coordinator=coordinator, runtime=Runtime(max_turns=40)),
        recorder,
    )


def write_session_trace(
    out_dir: Path,
    recorder: InMemoryRecorder,
    result: Result,
    wall_s: float,
    *,
    scenario: str = "console",
) -> Path:
    """Write the session's ``events.jsonl`` + ``manifest.json`` + ``digest.md``."""
    record = build_run_record(scenario, recorder.events, result, wall_s)
    return write_run(out_dir, record, recorder.events)


async def _converse(
    console: Console, recorder: InMemoryRecorder, session_dir: Path
) -> None:
    started = time.perf_counter()
    print(
        "agentique console [observed] — type a message; /artifacts, /approve <id>, "
        "/reject <id>, /quit"
    )
    print(f"# trace -> {session_dir}")

    def flush() -> None:
        if console.last_result is not None and recorder.events:
            write_session_trace(
                session_dir,
                recorder,
                console.last_result,
                time.perf_counter() - started,
            )

    try:
        while not console.done:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in {"/quit", "/exit"}:
                break
            try:
                if line == "/artifacts":
                    for artifact in await console.artifacts():
                        print(f"  {artifact.id} [{artifact.status}] {artifact.kind}")
                elif line.startswith("/approve "):
                    target = line.removeprefix("/approve ").strip()
                    artifact = await console.approve(target)
                    print(f"  {artifact.id} -> {artifact.status}")
                elif line.startswith("/reject "):
                    target = line.removeprefix("/reject ").strip()
                    artifact = await console.reject(target)
                    print(f"  {artifact.id} -> {artifact.status}")
                else:
                    turn = await console.send(line)
                    print(f"orchestrator> {turn.message}")
                    flush()
                    print(f"  [trace] {len(recorder.events)} events -> {session_dir}")
                    if turn.done:
                        print("[conversation ended]")
            except Exception as exc:  # keep the REPL alive on bad ids / API hiccups
                print(f"  error: {exc}")
    finally:
        flush()
        if console.last_result is not None and recorder.events:
            record = build_run_record(
                "console",
                recorder.events,
                console.last_result,
                time.perf_counter() - started,
            )
            print("\n" + render_digest(record, recorder.events))


def main() -> None:
    for key, value in load_env(Path(".env")).items():
        os.environ.setdefault(key, value)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (add it to .env or your environment).")
        return
    from agentique.anthropic import AnthropicModel

    model = AnthropicModel(os.environ.get("AGENTIQUE_MODEL", "claude-haiku-4-5"))
    console, recorder = wire(model)
    session_dir = _RUNS_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-console"
    asyncio.run(_converse(console, recorder, session_dir))


if __name__ == "__main__":
    main()
