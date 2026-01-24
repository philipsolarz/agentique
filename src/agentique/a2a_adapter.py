from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

from .models import AgentEvent, AgentResponse, McpContextSnapshot, StreamChunk


@dataclass(frozen=True)
class A2AImports:
    A2AClient: type | None
    ClientFactory: type | None
    ClientConfig: type | None
    create_text_message_object: Callable[..., Any] | None
    get_message_text: Callable[[Any], str] | None
    get_artifact_text: Callable[[Any], str] | None
    MessageSendParams: type | None
    MessageSendConfiguration: type | None
    SendMessageRequest: type | None
    SendStreamingMessageRequest: type | None


def _load_a2a_imports() -> A2AImports:
    try:
        import a2a  # noqa: F401
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "A2A SDK not installed. Install the A2A Python SDK to enable the bridge."
        ) from exc

    A2AClient = None
    ClientFactory = None
    ClientConfig = None
    create_text_message_object: Callable[..., Any] | None = None
    get_message_text: Callable[[Any], str] | None = None
    get_artifact_text: Callable[[Any], str] | None = None
    MessageSendParams = None
    MessageSendConfiguration = None
    SendMessageRequest = None
    SendStreamingMessageRequest = None

    try:
        from a2a.client import A2AClient
    except Exception:
        A2AClient = None

    try:
        from a2a.client import ClientFactory, ClientConfig
    except Exception:
        ClientFactory = None
        ClientConfig = None

    try:
        from a2a.client.helpers import create_text_message_object
    except Exception:
        create_text_message_object = None

    try:
        from a2a.utils.message import get_message_text
    except Exception:
        get_message_text = None

    try:
        from a2a.utils.artifact import get_artifact_text
    except Exception:
        get_artifact_text = None

    try:
        from a2a.types import MessageSendParams, SendMessageRequest
    except Exception:
        MessageSendParams = None
        SendMessageRequest = None

    try:
        from a2a.types import MessageSendConfiguration
    except Exception:
        MessageSendConfiguration = None

    try:
        from a2a.types import SendStreamingMessageRequest
    except Exception:
        SendStreamingMessageRequest = None

    return A2AImports(
        A2AClient=A2AClient,
        ClientFactory=ClientFactory,
        ClientConfig=ClientConfig,
        create_text_message_object=create_text_message_object,
        get_message_text=get_message_text,
        get_artifact_text=get_artifact_text,
        MessageSendParams=MessageSendParams,
        MessageSendConfiguration=MessageSendConfiguration,
        SendMessageRequest=SendMessageRequest,
        SendStreamingMessageRequest=SendStreamingMessageRequest,
    )


_A2A_IMPORTS: A2AImports | None = None


def _a2a() -> A2AImports:
    global _A2A_IMPORTS
    if _A2A_IMPORTS is None:
        _A2A_IMPORTS = _load_a2a_imports()
    return _A2A_IMPORTS


def _is_async_iterator(value: Any) -> bool:
    return hasattr(value, "__aiter__")


class A2AClientFactory:
    """Creates and caches A2A clients by base URL."""

    def __init__(
        self,
        *,
        prefer_legacy: bool = False,
        client_config: Any | None = None,
        resolver_http_kwargs: dict[str, Any] | None = None,
        card_path: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        self._prefer_legacy = prefer_legacy
        self._client_config = client_config
        self._resolver_http_kwargs = resolver_http_kwargs
        self._card_path = card_path
        self._client_kwargs = client_kwargs
        self._clients: dict[str, Any] = {}

    async def get(self, base_url: str) -> Any:
        if base_url in self._clients:
            return self._clients[base_url]

        imports = _a2a()
        client: Any | None = None

        if not self._prefer_legacy and imports.ClientFactory is not None:
            config = self._client_config
            if config is None and imports.ClientConfig is not None:
                config = imports.ClientConfig()
            client = await imports.ClientFactory.connect(
                base_url,
                client_config=config,
                resolver_http_kwargs=self._resolver_http_kwargs,
                relative_card_path=self._card_path,
            )
        elif imports.A2AClient is not None:
            client = imports.A2AClient(base_url, **self._client_kwargs)

        if client is None:
            raise RuntimeError("Unable to construct an A2A client with the installed SDK.")

        self._clients[base_url] = client
        return client

    async def aclose(self) -> None:
        for client in self._clients.values():
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        self._clients.clear()


class A2ATranslator:
    """Translate MCP requests into A2A messages and normalize responses."""

    def build_message(
        self,
        text: str,
        context: McpContextSnapshot | None,
        extra: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any]]:
        imports = _a2a()
        if imports.create_text_message_object is None:
            raise RuntimeError("A2A SDK missing create_text_message_object helper.")

        # create_text_message_object(role=Role.user, content='')
        # Pass text as content parameter (second positional arg)
        message = imports.create_text_message_object(content=text)
        metadata: dict[str, Any] = {}
        if context:
            metadata["mcp"] = context.to_metadata()
        if extra:
            metadata["extra"] = extra

        if metadata:
            message = self._with_metadata(message, metadata)

        return message, metadata

    def build_request(
        self,
        message: Any,
        *,
        metadata: dict[str, Any] | None,
        configuration: Any | None,
    ) -> Any:
        imports = _a2a()
        if imports.MessageSendParams is None or imports.SendMessageRequest is None:
            raise RuntimeError("A2A SDK missing legacy request types.")
        payload = imports.MessageSendParams(
            message=message,
            metadata=metadata,
            configuration=self._coerce_configuration(configuration),
        )
        request_id = str(uuid4())
        return imports.SendMessageRequest(id=request_id, params=payload)

    def build_streaming_request(
        self,
        message: Any,
        *,
        metadata: dict[str, Any] | None,
        configuration: Any | None,
    ) -> Any:
        imports = _a2a()
        if imports.SendStreamingMessageRequest is None:
            raise RuntimeError("Streaming request type unavailable in A2A SDK.")
        payload = imports.MessageSendParams(
            message=message,
            metadata=metadata,
            configuration=self._coerce_configuration(configuration),
        )
        request_id = str(uuid4())
        return imports.SendStreamingMessageRequest(id=request_id, params=payload)

    def iter_events(self, iterator: AsyncIterator[Any]) -> AsyncIterator[AgentEvent]:
        async def _iter() -> AsyncIterator[AgentEvent]:
            async for event in iterator:
                yield self.to_event(event)

        return _iter()

    def to_event(self, event: Any) -> AgentEvent:
        kind, text = self._event_to_text(event)
        return AgentEvent(kind=kind, text=text, raw=event)

    def reduce_events(self, events: list[AgentEvent]) -> str:
        text_parts = [event.text for event in events if event.text]
        return "".join(text_parts).strip()

    def event_to_chunk(self, agent: str, index: int, event: AgentEvent) -> StreamChunk:
        return StreamChunk(agent=agent, index=index, kind=event.kind, text=event.text)

    def _event_to_text(self, event: Any) -> tuple[str, str | None]:
        event = self._unwrap_result(event)
        if isinstance(event, tuple) and len(event) == 2:
            task, update = event
            if update is None:
                return "task", self._task_summary(task)
            status = getattr(update, "status", None)
            if status is not None:
                return "status", str(status)
            artifact = getattr(update, "artifact", None)
            if artifact is not None:
                text = self._artifact_text(artifact)
                return "artifact", text
        if hasattr(event, "parts"):
            return "message", self._message_text(event)
        status = getattr(event, "status", None)
        if status is not None:
            return "status", str(status)
        return "event", None

    def _unwrap_result(self, event: Any) -> Any:
        if hasattr(event, "result"):
            return getattr(event, "result")
        return event

    def _message_text(self, message: Any) -> str | None:
        imports = _a2a()
        if imports.get_message_text is not None:
            try:
                return imports.get_message_text(message)
            except Exception:
                pass
        return self._extract_text_from_parts(getattr(message, "parts", None))

    def _artifact_text(self, artifact: Any) -> str | None:
        imports = _a2a()
        if imports.get_artifact_text is not None:
            try:
                return imports.get_artifact_text(artifact)
            except Exception:
                pass
        return self._extract_text_from_parts(getattr(artifact, "parts", None))

    def _extract_text_from_parts(self, parts: Any) -> str | None:
        if not parts:
            return None
        texts: list[str] = []
        for part in parts:
            if isinstance(part, str):
                texts.append(part)
                continue
            if isinstance(part, dict) and "text" in part:
                value = part.get("text")
                if isinstance(value, str):
                    texts.append(value)
                continue
            text_attr = getattr(part, "text", None)
            if isinstance(text_attr, str):
                texts.append(text_attr)
        if not texts:
            return None
        return "".join(texts)

    def _with_metadata(self, message: Any, metadata: dict[str, Any]) -> Any:
        if hasattr(message, "model_copy"):
            return message.model_copy(update={"metadata": metadata})
        if hasattr(message, "copy"):
            return message.copy(update={"metadata": metadata})
        if hasattr(message, "metadata"):
            try:
                setattr(message, "metadata", metadata)
                return message
            except Exception:
                return message
        return message

    def _coerce_configuration(self, configuration: Any | None) -> Any | None:
        if configuration is None:
            return None
        imports = _a2a()
        if imports.MessageSendConfiguration is not None and isinstance(configuration, dict):
            try:
                return imports.MessageSendConfiguration(**configuration)
            except Exception:
                return configuration
        return configuration

    def _task_summary(self, task: Any) -> str | None:
        task_id = getattr(task, "id", None) or getattr(task, "task_id", None)
        status = getattr(task, "status", None)
        if task_id and status:
            return f"{task_id}: {status}"
        if status:
            return str(status)
        return None


class A2ABridge:
    def __init__(self, client_factory: A2AClientFactory, translator: A2ATranslator) -> None:
        self._client_factory = client_factory
        self._translator = translator

    async def send_message(
        self,
        base_url: str,
        text: str,
        *,
        context: McpContextSnapshot | None = None,
        metadata: dict[str, Any] | None = None,
        configuration: Any | None = None,
    ) -> AgentResponse:
        client = await self._get_client(base_url)
        message, request_metadata = self._translator.build_message(text, context, metadata)

        events: list[AgentEvent] = []
        async for event in self._iter_events(
            client,
            message,
            request_metadata=request_metadata,
            configuration=configuration,
            streaming=False,
        ):
            events.append(event)

        return AgentResponse(
            agent=base_url,
            text=self._translator.reduce_events(events),
            events=tuple(events),
        )

    async def stream_message(
        self,
        base_url: str,
        text: str,
        *,
        context: McpContextSnapshot | None = None,
        metadata: dict[str, Any] | None = None,
        configuration: Any | None = None,
    ) -> AsyncIterator[AgentEvent]:
        client = await self._get_client(base_url)
        message, request_metadata = self._translator.build_message(text, context, metadata)

        async for event in self._iter_events(
            client,
            message,
            request_metadata=request_metadata,
            configuration=configuration,
            streaming=True,
        ):
            yield event

    async def get_agent_card(self, base_url: str) -> Any | None:
        client = await self._get_client(base_url)
        getter = getattr(client, "get_card", None)
        if callable(getter):
            result = getter()
            if inspect.isawaitable(result):
                return await result
            return result
        return None

    async def _iter_events(
        self,
        client: Any,
        message: Any,
        *,
        request_metadata: dict[str, Any],
        configuration: Any | None,
        streaming: bool,
    ) -> AsyncIterator[AgentEvent]:
        if hasattr(client, "send_message_streaming"):
            if streaming:
                request = self._translator.build_streaming_request(
                    message,
                    metadata=request_metadata,
                    configuration=configuration,
                )
                iterator = client.send_message_streaming(request)
                async for event in self._translator.iter_events(iterator):
                    yield event
                return

            request = self._translator.build_request(
                message,
                metadata=request_metadata,
                configuration=configuration,
            )
            response = await client.send_message(request)
            yield self._translator.to_event(response)
            return

        try:
            result = client.send_message(
                message,
                configuration=self._translator._coerce_configuration(configuration),
                request_metadata=request_metadata,
            )
        except TypeError:
            request = self._translator.build_request(
                message,
                metadata=request_metadata,
                configuration=configuration,
            )
            response = await client.send_message(request)
            yield self._translator.to_event(response)
            return

        if _is_async_iterator(result):
            async for event in result:
                yield self._translator.to_event(event)
            return

        if inspect.isawaitable(result):
            response = await result
            yield self._translator.to_event(response)
            return

        yield self._translator.to_event(result)

    async def _get_client(self, base_url: str) -> Any:
        result = self._client_factory.get(base_url)
        if inspect.isawaitable(result):
            return await result
        return result

    async def aclose(self) -> None:
        await self._client_factory.aclose()
