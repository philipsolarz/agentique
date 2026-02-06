"""Mock implementations for testing agentique.

These classes satisfy the ``AgentAdapter``, ``BridgeMiddleware``, and
related protocols via structural subtyping, enabling unit and
integration tests without real agent backends.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import uuid4

from agentique.core.errors import AgentNotFoundError
from agentique.core.types import (
    AgentEvent,
    AgentInfo,
    AgentResponse,
    BridgeContext,
)


# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------


class MockAdapter:
    """In-memory adapter that returns pre-configured responses.

    Satisfies the ``AgentAdapter`` protocol.

    Usage::

        adapter = MockAdapter(responses={"calc": "42"})
        resp = await adapter.send_message("calc", "what is 6*7?", ctx)
        assert resp.text == "42"

        # Also records all calls
        assert len(adapter.calls) == 1
        assert adapter.calls[0]["message"] == "what is 6*7?"
    """

    def __init__(
        self,
        agents: dict[str, AgentInfo] | None = None,
        responses: dict[str, str] | None = None,
        *,
        default_response: str = "Mock response",
        delay: float = 0.0,
    ) -> None:
        self._agents = agents or {}
        self._responses = responses or {}
        self._default = default_response
        self._delay = delay
        self.calls: list[dict[str, Any]] = []
        self._closed = False

    async def discover_agents(self) -> list[AgentInfo]:
        return list(self._agents.values())

    async def send_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AgentResponse:
        self._record(agent_id, message, context)
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        text = self._responses.get(agent_id, self._default)
        event = AgentEvent(kind="message", text=text, task_id=str(uuid4()))
        return AgentResponse(agent=agent_id, text=text, events=(event,))

    async def stream_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AsyncIterator[AgentEvent]:
        self._record(agent_id, message, context)
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        text = self._responses.get(agent_id, self._default)
        yield AgentEvent(kind="message", text=text, task_id=str(uuid4()))

    async def get_agent_card(self, agent_id: str) -> dict[str, Any] | None:
        info = self._agents.get(agent_id)
        if info is None:
            return None
        return mock_agent_card(info.name, skills=list(info.skills))

    async def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def _record(self, agent_id: str, message: str, context: BridgeContext) -> None:
        self.calls.append({
            "agent_id": agent_id,
            "message": message,
            "session_id": context.session_id,
            "request_id": context.request_id,
        })


class MockStreamingAdapter(MockAdapter):
    """Mock adapter that streams responses word-by-word.

    Useful for testing streaming behaviour and progress tracking.
    """

    def __init__(
        self,
        agents: dict[str, AgentInfo] | None = None,
        responses: dict[str, str] | None = None,
        *,
        default_response: str = "Mock streaming response",
        chunk_delay: float = 0.01,
    ) -> None:
        super().__init__(
            agents=agents,
            responses=responses,
            default_response=default_response,
        )
        self._chunk_delay = chunk_delay

    async def stream_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AsyncIterator[AgentEvent]:
        self._record(agent_id, message, context)
        text = self._responses.get(agent_id, self._default)
        words = text.split()
        task_id = str(uuid4())

        for i, word in enumerate(words):
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            progress = ((i + 1) / len(words)) * 100
            yield AgentEvent(
                kind="message",
                text=word + (" " if i < len(words) - 1 else ""),
                task_id=task_id,
                progress=progress,
                is_final=(i == len(words) - 1),
            )


# ---------------------------------------------------------------------------
# Mock middleware
# ---------------------------------------------------------------------------


class RecordingMiddleware:
    """Middleware that records all requests passing through it.

    Satisfies the ``BridgeMiddleware`` protocol.

    Usage::

        mw = RecordingMiddleware()
        chain = MiddlewareChain()
        chain.add(mw)
        await chain.execute(request, handler)
        assert len(mw.requests) == 1
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[Any] = []

    async def process(
        self,
        request: dict[str, Any],
        call_next: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        self.requests.append(dict(request))
        result = await call_next(request)
        self.responses.append(result)
        return result


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def mock_agent_info(
    name: str = "test-agent",
    *,
    base_url: str = "http://localhost:9999",
    skills: tuple[str, ...] | list[str] = (),
    description: str | None = None,
) -> AgentInfo:
    """Create a mock ``AgentInfo`` for testing."""
    return AgentInfo(
        name=name,
        base_url=base_url,
        description=description or f"Test agent: {name}",
        skills=tuple(skills),
    )


def mock_agent_card(
    name: str = "test-agent",
    *,
    skills: list[str] | None = None,
    sub_agents: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a mock A2A agent card dictionary."""
    card: dict[str, Any] = {
        "name": name,
        "description": f"Mock agent card for {name}",
        "version": "1.0.0",
        "url": f"http://localhost:9999",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
    }

    if skills:
        card["skills"] = [
            {"id": s, "name": s, "description": f"Skill: {s}"}
            for s in skills
        ]

    metadata: dict[str, Any] = {"author": "agentique-testing"}
    if sub_agents:
        metadata["sub_agents"] = sub_agents
    card["metadata"] = metadata

    if tools:
        card["capabilities"]["extensions"] = [
            {
                "uri": "urn:mcp:extension:tools",
                "params": {"mcpTools": tools},
            }
        ]

    return card
