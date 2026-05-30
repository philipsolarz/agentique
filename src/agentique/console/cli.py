"""The REPL: a thin shell over Console for talking to the orchestrator.

Reads ``ANTHROPIC_API_KEY`` from the environment (or a local ``.env`` parsed with
the standard library — no third-party loader), builds a Console over the Anthropic
provider, and loops: operator text goes to the orchestrator; ``/artifacts``,
``/approve <id>``, ``/reject <id>`` manage artifacts; ``/quit`` exits.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agentique.console.console import Console, build_console

# Anthropic's fast tier; override with AGENTIQUE_MODEL. There is deliberately no
# library-wide default model id — the console picks one because it is an app.
_DEFAULT_MODEL = "claude-haiku-4-5"


def load_env(path: Path) -> dict[str, str]:
    """Parse a minimal ``.env`` (``KEY=VALUE`` per line) with the standard library.

    Blank lines and ``#`` comments are ignored; surrounding single/double quotes on
    the value are stripped. No third-party dependency, keeping import discipline
    intact.
    """
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _build_console() -> Console:
    from agentique.anthropic import AnthropicModel  # lazy: only here is the SDK touched

    model_id = os.environ.get("AGENTIQUE_MODEL", _DEFAULT_MODEL)
    return build_console(AnthropicModel(model_id))


async def _converse(console: Console) -> None:
    print(
        "agentique console — the orchestrator dispatches a fleet (planner, "
        "explorer, builder, reviewer)."
    )
    print(
        "type a message; /artifacts, /approve <id>, /reject <id>, /quit. "
        "Built files land under ./workspace/."
    )
    while not console.done:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            return
        try:
            if line == "/artifacts":
                for artifact in await console.artifacts():
                    preview = artifact.payload[:60].replace("\n", " ")
                    print(
                        f"  {artifact.id} [{artifact.status}] "
                        f"{artifact.kind}: {preview}"
                    )
            elif line.startswith("/approve "):
                artifact = await console.approve(line.removeprefix("/approve ").strip())
                print(f"  {artifact.id} -> {artifact.status}")
            elif line.startswith("/reject "):
                artifact = await console.reject(line.removeprefix("/reject ").strip())
                print(f"  {artifact.id} -> {artifact.status}")
            else:
                turn = await console.send(line)
                print(f"orchestrator> {turn.message}")
                if turn.done:
                    print("[conversation ended]")
        except Exception as exc:  # keep the REPL alive on bad ids / API hiccups
            print(f"  error: {exc}")


def main() -> None:
    for key, value in load_env(Path(".env")).items():
        os.environ.setdefault(key, value)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (add it to .env or your environment).")
        return
    asyncio.run(_converse(_build_console()))
