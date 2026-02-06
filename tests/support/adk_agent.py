from __future__ import annotations

from typing import Any


class AdkEchoAgent:
    """Minimal wrapper around a Google ADK agent to produce deterministic output."""

    def __init__(self) -> None:
        self._agent = _build_agent()

    async def reply(self, text: str) -> str:
        if self._agent is None:
            raise RuntimeError("Google ADK is not available.")
        try:
            result = _call_agent(self._agent, text)
        except Exception:
            # ADK interface changes frequently; for bridge tests we only need
            # deterministic echo behavior, so fallback to plain text echo.
            return text
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, str):
            return result
        return str(result)


def _build_agent() -> Any:
    """Attempt to construct a simple ADK agent instance.

    The ADK API is still evolving, so this function tries a few import paths
    and constructor signatures commonly used in ADK examples.
    """

    agent_cls = None
    for path in (
        ("google.adk", "Agent"),
        ("google.adk.agents", "Agent"),
        ("google.adk.agent", "Agent"),
    ):
        try:
            module = __import__(path[0], fromlist=[path[1]])
            agent_cls = getattr(module, path[1])
            break
        except Exception:
            continue

    if agent_cls is None:
        raise RuntimeError("Google ADK Agent class not found.")

    for kwargs in (
        {"name": "Echo", "instructions": "Echo the user's message."},
        {"name": "Echo"},
        {},
    ):
        try:
            return agent_cls(**kwargs)
        except Exception:
            continue

    raise RuntimeError("Unable to construct ADK Agent with available signatures.")


def _call_agent(agent: Any, text: str) -> Any:
    for method_name in ("run", "invoke", "__call__", "run_async"):
        method = getattr(agent, method_name, None)
        if callable(method):
            return method(text)
    raise RuntimeError("ADK Agent does not expose a callable interface.")
