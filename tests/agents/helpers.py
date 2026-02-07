"""Helper functions for creating testable A2A applications from mock agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
from a2a.client.client_factory import minimal_agent_card
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

if TYPE_CHECKING:
    from a2a.server.agent_execution import AgentExecutor


def create_mock_a2a_app(
    executor: AgentExecutor,
    name: str = "test-agent",
    streaming: bool = True,
    push_notifications: bool = False,
    skills: list[AgentSkill] | None = None,
) -> Any:
    """Build a testable A2A Starlette app from an AgentExecutor.

    Args:
        executor: The agent executor implementation
        name: Agent name for the card
        streaming: Whether the agent supports streaming
        push_notifications: Whether the agent supports push notifications
        skills: Optional list of skills; defaults to a single test skill

    Returns:
        Starlette application ready for testing via ASGITransport
    """
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    if skills is None:
        skills = [AgentSkill(id="test", name="Test", description="Test skill", tags=["test"])]

    card = AgentCard(
        name=name,
        description=f"Test agent: {name}",
        url="http://testserver/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=streaming, pushNotifications=push_notifications),
        skills=skills,
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
    )

    builder = A2AStarletteApplication(agent_card=card, http_handler=handler)
    return builder.build()


async def create_a2a_client_for_executor(executor: AgentExecutor, name: str = "test-agent") -> httpx.AsyncClient:
    """Create an httpx AsyncClient connected to a mock A2A agent via ASGITransport.

    This avoids the need for a real HTTP server - the agent runs in-process.

    Args:
        executor: The agent executor implementation
        name: Agent name

    Returns:
        httpx.AsyncClient configured with ASGITransport
    """
    app = create_mock_a2a_app(executor, name=name)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")
