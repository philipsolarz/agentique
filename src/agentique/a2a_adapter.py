from __future__ import annotations

try:
    from a2a.client import A2AClient, ClientFactory, ClientConfig
    from a2a.client.helpers import create_text_message_object
    from a2a.utils.message import get_message_text
    from a2a.utils.artifact import get_artifact_text
    from a2a.types import (
        MessageSendParams,
        SendMessageRequest,
        MessageSendConfiguration,
        SendStreamingMessageRequest,
    )
except ImportError as exc:
    raise RuntimeError(
        "A2A SDK not installed or missing required components. "
        "Install a compatible version of 'a2a-sdk'."
    ) from exc

import inspect
from typing import Any, AsyncIterator
from uuid import uuid4

from .models import AgentEvent, AgentResponse, McpContextSnapshot, StreamChunk


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

        client: Any | None = None

        if not self._prefer_legacy:
            config = self._client_config
            if config is None:
                # Create default config with extended timeout for multi-turn agent operations
                # Default httpx timeout is 5s, but agents with function calling need more time
                import httpx
                httpx_client = httpx.AsyncClient(timeout=60.0)  # 60 seconds for agent operations
                config = ClientConfig(httpx_client=httpx_client)
            client = await ClientFactory.connect(
                base_url,
                client_config=config,
                resolver_http_kwargs=self._resolver_http_kwargs,
                relative_card_path=self._card_path,
            )
        else:
            client = A2AClient(base_url, **self._client_kwargs)

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
        # create_text_message_object(role=Role.user, content='')
        # Pass text as content parameter (second positional arg)
        message = create_text_message_object(content=text)
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
        payload = MessageSendParams(
            message=message,
            metadata=metadata,
            configuration=self._coerce_configuration(configuration),
        )
        request_id = str(uuid4())
        return SendMessageRequest(id=request_id, params=payload)

    def build_streaming_request(
        self,
        message: Any,
        *,
        metadata: dict[str, Any] | None,
        configuration: Any | None,
    ) -> Any:
        payload = MessageSendParams(
            message=message,
            metadata=metadata,
            configuration=self._coerce_configuration(configuration),
        )
        request_id = str(uuid4())
        return SendStreamingMessageRequest(id=request_id, params=payload)

    def iter_events(self, iterator: AsyncIterator[Any]) -> AsyncIterator[AgentEvent]:
        async def _iter() -> AsyncIterator[AgentEvent]:
            async for event in iterator:
                yield self.to_event(event)

        return _iter()

    def to_event(self, event: Any) -> AgentEvent:
        kind, text = self._event_to_text(event)
        metadata = self._extract_full_event_metadata(event)
        return AgentEvent(
            kind=kind,
            text=text,
            raw=event,
            task_id=metadata.get("task_id"),
            context_id=metadata.get("context_id"),
            progress=metadata.get("progress"),
            artifact_id=metadata.get("artifact_id"),
            artifact_name=metadata.get("artifact_name"),
            branch=metadata.get("branch"),
            author=metadata.get("author"),
            state=metadata.get("state"),
            is_final=metadata.get("is_final", False),
            requires_confirmation=metadata.get("requires_confirmation", False),
            tool_call=metadata.get("tool_call"),
            event_metadata=metadata.get("event_metadata"),
        )

    def _extract_full_event_metadata(self, event: Any) -> dict[str, Any]:
        """Extract comprehensive metadata from A2A events including sub-agent info."""
        result: dict[str, Any] = {}
        event = self._unwrap_result(event)

        # Handle tuple events (task, update)
        if isinstance(event, tuple) and len(event) == 2:
            task, update = event
            result["task_id"] = getattr(task, "id", None) or getattr(task, "task_id", None)
            result["context_id"] = getattr(task, "context_id", None)

            # Extract status/state
            status = getattr(task, "status", None)
            if status:
                state = getattr(status, "state", None)
                if state:
                    result["state"] = str(state.value) if hasattr(state, "value") else str(state)
                result["is_final"] = getattr(update, "final", False) if update else False

            if update:
                # Progress
                progress_val = getattr(update, "progress", None)
                if progress_val is not None and isinstance(progress_val, (int, float)):
                    result["progress"] = float(progress_val)

                # Artifact
                artifact = getattr(update, "artifact", None)
                if artifact:
                    result["artifact_id"] = getattr(artifact, "id", None) or getattr(artifact, "artifact_id", None)
                    result["artifact_name"] = getattr(artifact, "name", None)

            # Task metadata (may contain branch info from ADK)
            task_metadata = getattr(task, "metadata", None)
            if task_metadata and isinstance(task_metadata, dict):
                result["event_metadata"] = dict(task_metadata)
                # Extract ADK-specific fields
                result["branch"] = task_metadata.get("branch")
                result["author"] = task_metadata.get("author")

        # Handle direct message/event
        else:
            result["task_id"] = getattr(event, "task_id", None)
            result["context_id"] = getattr(event, "context_id", None)

            # Check for branch/author in message metadata (ADK pattern)
            event_metadata = getattr(event, "metadata", None)
            if event_metadata and isinstance(event_metadata, dict):
                result["event_metadata"] = dict(event_metadata)
                result["branch"] = event_metadata.get("branch")
                result["author"] = event_metadata.get("author")

                # Check for tool confirmation request
                if event_metadata.get("requires_confirmation"):
                    result["requires_confirmation"] = True
                    result["tool_call"] = event_metadata.get("tool_call")

        return result

    def reduce_events(self, events: list[AgentEvent]) -> str:
        text_parts = [event.text for event in events if event.text]
        return "".join(text_parts).strip()

    def event_to_chunk(self, agent: str, index: int, event: AgentEvent) -> StreamChunk:
        return StreamChunk.from_event(agent, index, event)

    def _event_to_text(self, event: Any) -> tuple[str, str | None]:
        event = self._unwrap_result(event)
        if isinstance(event, tuple) and len(event) == 2:
            task, update = event
            if update is None:
                # Check if the task itself has a message/result
                # This happens with completed tasks from ADK's to_a2a wrapper
                task_message = getattr(task, "message", None)
                if task_message is not None:
                    text = self._message_text(task_message)
                    if text:
                        return "message", text
                # Check for result field on task
                task_result = getattr(task, "result", None)
                if task_result is not None:
                    if hasattr(task_result, "parts"):
                        text = self._message_text(task_result)
                        if text:
                            return "message", text
                # Fall back to task summary if no message found
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
        try:
            return get_message_text(message)
        except Exception:
            pass
        return self._extract_text_from_parts(getattr(message, "parts", None))

    def _artifact_text(self, artifact: Any) -> str | None:
        try:
            return get_artifact_text(artifact)
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
        if isinstance(configuration, dict):
            try:
                return MessageSendConfiguration(**configuration)
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

    def _extract_event_metadata(self, event: Any) -> tuple[str | None, float | None, str | None, dict[str, Any] | None]:
        """Extract structured metadata from A2A events.

        Returns:
            Tuple of (task_id, progress, artifact_id, metadata)
        """
        task_id = None
        progress = None
        artifact_id = None
        metadata = None

        event = self._unwrap_result(event)

        # Extract task information
        if isinstance(event, tuple) and len(event) == 2:
            task, update = event
            task_id = getattr(task, "id", None) or getattr(task, "task_id", None)

            # Extract progress if available
            if update:
                progress_val = getattr(update, "progress", None)
                if progress_val is not None and isinstance(progress_val, (int, float)):
                    progress = float(progress_val)

                # Extract artifact information
                artifact = getattr(update, "artifact", None)
                if artifact:
                    artifact_id = getattr(artifact, "id", None) or getattr(artifact, "artifact_id", None)

            # Extract task metadata
            task_metadata = getattr(task, "metadata", None)
            if task_metadata and isinstance(task_metadata, dict):
                metadata = dict(task_metadata)

        # Extract message/event metadata
        event_metadata = getattr(event, "metadata", None)
        if event_metadata and isinstance(event_metadata, dict):
            if metadata:
                metadata.update(event_metadata)
            else:
                metadata = dict(event_metadata)

        return task_id, progress, artifact_id, metadata


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
        try:
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
        except Exception as exc:
            raise RuntimeError(f"Failed to send message to A2A agent at {base_url}: {exc}") from exc

    async def stream_message(
        self,
        base_url: str,
        text: str,
        *,
        context: McpContextSnapshot | None = None,
        metadata: dict[str, Any] | None = None,
        configuration: Any | None = None,
    ) -> AsyncIterator[AgentEvent]:
        try:
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
        except Exception as exc:
            raise RuntimeError(f"Failed to stream message to A2A agent at {base_url}: {exc}") from exc

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
