"""Part 2 — the first live walk.

A real Agent + AnthropicModel + the ReadFile tool, wrapped in the recording layer
and driven through the real Engine: read a file and summarize it. The first time
the whole stack runs against the live API rather than a stub.

Requires .env with ANTHROPIC_API_KEY and ANTHROPIC_MODEL (see .env.example).

    uv run --extra anthropic python -m scenarios.walk
"""

from __future__ import annotations

import asyncio

from agentique.anthropic import AnthropicModel
from agentique.core import Agent
from agentique.tools import ReadFile
from scenarios.driver import run_scenario
from scenarios.env import load_env, require_env

_INSTRUCTIONS = (
    "You are a careful assistant. When the user asks about a file, use the "
    "read_file tool to read it before answering."
)
_PROMPT = "Read the file README.md and summarize it in 3 bullet points."


async def main() -> None:
    load_env()
    require_env("ANTHROPIC_API_KEY")  # read automatically by the Anthropic SDK
    model_id = require_env("ANTHROPIC_MODEL")

    agent = Agent(
        name="reader",
        instructions=_INSTRUCTIONS,
        model=AnthropicModel(model=model_id),
        tools=(ReadFile(),),
    )

    run = await run_scenario("walk", agent, _PROMPT)

    print(f"outcome: {type(run.result).__name__}")
    print(f"turns: {run.record.turns}  tool calls: {run.record.tool_calls}")
    print(f"artifacts: {run.out_dir}")
    print()
    print((run.out_dir / "digest.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
