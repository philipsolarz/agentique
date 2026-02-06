from __future__ import annotations

from typing import Any, AsyncIterator, Iterable

from a2a.client.client_factory import minimal_agent_card
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.types import Message
from a2a.utils.errors import MethodNotImplementedError
from a2a.utils.message import get_message_text, new_agent_text_message

from tests.support.adk_agent import AdkEchoAgent


class AdkRequestHandler(RequestHandler):
    def __init__(self, agent: AdkEchoAgent) -> None:
        self._agent = agent
        self.last_metadata: dict[str, Any] | None = None

    async def on_message_send(self, request: Any, context: Any | None = None) -> Message:
        params = _extract_send_params(request)
        self.last_metadata = getattr(params, "metadata", None) or getattr(
            params.message, "metadata", None
        )
        text = get_message_text(params.message)
        response_text = await self._agent.reply(text)
        return new_agent_text_message(response_text)

    def on_message_send_stream(
        self, request: Any, context: Any | None = None
    ) -> AsyncIterator[Message]:
        async def _stream() -> AsyncIterator[Message]:
            params = _extract_send_params(request)
            self.last_metadata = getattr(params, "metadata", None) or getattr(
                params.message, "metadata", None
            )
            text = get_message_text(params.message)
            response_text = await self._agent.reply(text)
            for chunk in _chunk_text(response_text, size=32):
                yield new_agent_text_message(chunk)

        return _stream()

    async def on_get_task(self, params: Any, context: Any | None = None) -> Any:
        raise MethodNotImplementedError("tasks/get not supported in test agent")

    async def on_cancel_task(self, params: Any, context: Any | None = None) -> Any:
        raise MethodNotImplementedError("tasks/cancel not supported in test agent")

    async def on_resubscribe_to_task(
        self, request: Any, context: Any | None = None
    ) -> AsyncIterator[Message]:
        raise MethodNotImplementedError("tasks/resubscribe not supported in test agent")

    async def on_set_task_push_notification_config(self, params: Any, context: Any | None = None) -> Any:
        raise MethodNotImplementedError("pushNotificationConfig/set not supported in test agent")

    async def on_get_task_push_notification_config(self, params: Any, context: Any | None = None) -> Any:
        raise MethodNotImplementedError("pushNotificationConfig/get not supported in test agent")

    async def on_list_task_push_notification_config(self, params: Any, context: Any | None = None) -> Any:
        raise MethodNotImplementedError("pushNotificationConfig/list not supported in test agent")

    async def on_delete_task_push_notification_config(self, params: Any, context: Any | None = None) -> None:
        raise MethodNotImplementedError("pushNotificationConfig/delete not supported in test agent")


def create_a2a_app(agent: AdkEchoAgent, base_url: str) -> tuple[Any, AdkRequestHandler]:
    card = minimal_agent_card(base_url)
    if hasattr(card, "name"):
        card.name = "adk-test-agent"
    if hasattr(card, "description"):
        card.description = "Google ADK test agent"
    if hasattr(card, "capabilities") and hasattr(card.capabilities, "streaming"):
        card.capabilities.streaming = True

    handler = AdkRequestHandler(agent)
    app = A2AFastAPIApplication(agent_card=card, http_handler=handler)
    return app.build(), handler


def _chunk_text(text: str, *, size: int) -> Iterable[str]:
    if not text:
        return [""]
    return (text[i : i + size] for i in range(0, len(text), size))


def _extract_send_params(request: Any) -> Any:
    """Support both legacy ``SendMessageRequest`` and modern ``MessageSendParams``."""
    return getattr(request, "params", request)
