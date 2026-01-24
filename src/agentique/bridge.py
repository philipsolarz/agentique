from __future__ import annotations

from typing import Any, AsyncIterator

from .a2a_adapter import A2ABridge, A2AClientFactory, A2ATranslator
from .models import AgentEvent, AgentResponse, McpContextSnapshot, StreamChunk
from .router import AgentRouter


class RouterBridge:
    """Coordinates routing, context propagation, and A2A interaction."""

    def __init__(
        self,
        router: AgentRouter,
        *,
        client_factory: A2AClientFactory | None = None,
        translator: A2ATranslator | None = None,
    ) -> None:
        self._router = router
        self._translator = translator or A2ATranslator()
        self._a2a = A2ABridge(client_factory or A2AClientFactory(), self._translator)

    async def send(
        self,
        text: str,
        *,
        ctx: Any,
        agent: str | None = None,
        skill: str | None = None,
        metadata: dict[str, Any] | None = None,
        configuration: Any | None = None,
    ) -> AgentResponse:
        descriptor = self._router.resolve(name=agent, skill=skill)
        snapshot = McpContextSnapshot.from_context(ctx)
        response = await self._a2a.send_message(
            descriptor.base_url,
            text,
            context=snapshot,
            metadata=metadata,
            configuration=configuration,
        )
        return AgentResponse(
            agent=descriptor.name,
            text=response.text,
            events=response.events,
            metadata={"agent": descriptor.to_dict()},
        )

    async def stream(
        self,
        text: str,
        *,
        ctx: Any,
        agent: str | None = None,
        skill: str | None = None,
        metadata: dict[str, Any] | None = None,
        configuration: Any | None = None,
    ) -> AsyncIterator[StreamChunk]:
        descriptor = self._router.resolve(name=agent, skill=skill)
        snapshot = McpContextSnapshot.from_context(ctx)
        index = 0
        async for event in self._a2a.stream_message(
            descriptor.base_url,
            text,
            context=snapshot,
            metadata=metadata,
            configuration=configuration,
        ):
            chunk = self._translator.event_to_chunk(descriptor.name, index, event)
            index += 1
            yield chunk

    async def get_agent_card(self, agent: str) -> Any | None:
        descriptor = self._router.describe(agent)
        return await self._a2a.get_agent_card(descriptor.base_url)

    async def aclose(self) -> None:
        await self._a2a.aclose()
