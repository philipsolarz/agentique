"""A2A agent adapter implementing the ``AgentAdapter`` protocol.

This is the primary adapter for communicating with agents that speak
the A2A protocol. It translates between agentique's core types and the
A2A SDK's message/request types.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
from typing import Any, AsyncIterator, Sequence
from uuid import uuid4

from agentique.core.types import (
    AgentEvent,
    AgentInfo,
    AgentResponse,
    BridgeContext,
)
from .client import A2AClientPool
from .card_parser import A2ACardParser

logger = logging.getLogger(__name__)

try:
    from a2a.client.helpers import create_text_message_object
    from a2a.utils.message import get_message_text
    from a2a.utils.artifact import get_artifact_text
    from a2a.types import (
        MessageSendConfiguration,
        MessageSendParams,
        PushNotificationConfig,
        SendMessageRequest,
        SendStreamingMessageRequest,
        TaskIdParams,
    )
except ImportError as exc:
    raise RuntimeError(
        "a2a-sdk is required for the A2A adapter. "
        "Install with: pip install 'agentique[a2a]'"
    ) from exc

_STREAM_RETRY_HINTS = (
    "timeout",
    "connection",
    "disconnect",
    "broken",
    "reset",
    "eof",
    "network",
    "stream",
)


@dataclass(frozen=True)
class _SendOptions:
    configuration: MessageSendConfiguration | None = None
    extensions: tuple[str, ...] = ()
    max_resubscribe_attempts: int = 1


class A2AAgentAdapter:
    """Adapter that bridges agentique to A2A-protocol agents.

    Implements the ``AgentAdapter`` protocol via structural subtyping.
    """

    def __init__(
        self,
        agents: dict[str, AgentInfo],
        *,
        client_pool: A2AClientPool | None = None,
        card_parser: A2ACardParser | None = None,
        default_extensions: Sequence[str] | None = None,
        default_push_notification: PushNotificationConfig | dict[str, Any] | None = None,
        max_resubscribe_attempts: int = 1,
    ) -> None:
        self._agents = dict(agents)
        self._pool = client_pool or A2AClientPool()
        self._parser = card_parser or A2ACardParser()
        self._default_extensions = tuple(default_extensions or ())
        self._default_push_notification = self._coerce_push_config(
            default_push_notification,
        )
        self._max_resubscribe_attempts = max(max_resubscribe_attempts, 0)

    # ---- AgentAdapter protocol ----

    async def discover_agents(self) -> list[AgentInfo]:
        return list(self._agents.values())

    async def send_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AgentResponse:
        info = self._require_agent(agent_id)
        client = await self._pool.get(info.base_url)
        msg_obj, metadata, options = self._build_message(message, context)

        events: list[AgentEvent] = []
        async for event in self._iter_events(
            client, msg_obj, metadata=metadata, options=options, streaming=False,
        ):
            events.append(event)

        text = "".join(e.text for e in events if e.text).strip()
        return AgentResponse(
            agent=agent_id, text=text, events=tuple(events),
        )

    async def stream_message(
        self,
        agent_id: str,
        message: str,
        context: BridgeContext,
    ) -> AsyncIterator[AgentEvent]:
        info = self._require_agent(agent_id)
        client = await self._pool.get(info.base_url)
        msg_obj, metadata, options = self._build_message(message, context)

        async for event in self._iter_events(
            client, msg_obj, metadata=metadata, options=options, streaming=True,
        ):
            yield event

    async def get_agent_card(self, agent_id: str) -> Any | None:
        """Fetch the raw agent card for *agent_id*."""
        info = self._require_agent(agent_id)
        client = await self._pool.get(info.base_url)
        getter = getattr(client, "get_card", None)
        if callable(getter):
            result = getter(extensions=list(self._default_extensions) or None)
            if inspect.isawaitable(result):
                return await result
            return result
        getter = getattr(client, "get_agent_card", None)
        if callable(getter):
            result = getter()
            return await result if inspect.isawaitable(result) else result
        return None

    async def close(self) -> None:
        await self._pool.close()

    # ---- internal helpers ----

    def _require_agent(self, agent_id: str) -> AgentInfo:
        info = self._agents.get(agent_id)
        if info is None:
            from agentique.core.errors import AgentNotFoundError
            raise AgentNotFoundError(
                f"Unknown agent '{agent_id}'. "
                f"Available: {sorted(self._agents)}"
            )
        return info

    def _build_message(
        self, text: str, context: BridgeContext,
    ) -> tuple[Any, dict[str, Any], _SendOptions]:
        msg = create_text_message_object(content=text)
        metadata: dict[str, Any] = {"mcp": context.to_metadata()}
        if context.conversation_history:
            metadata["conversation_history"] = context.conversation_history
        options = self._build_send_options(context)
        # Attach metadata to message
        if hasattr(msg, "model_copy"):
            msg = msg.model_copy(update={"metadata": metadata})
        elif hasattr(msg, "metadata"):
            try:
                msg.metadata = metadata
            except Exception:
                pass
        return msg, metadata, options

    def _build_send_options(self, context: BridgeContext) -> _SendOptions:
        meta = context.meta if isinstance(context.meta, dict) else {}
        a2a_meta = meta.get("a2a") if isinstance(meta.get("a2a"), dict) else {}
        extensions = list(self._default_extensions)
        extensions.extend(_str_list(a2a_meta.get("extensions")))
        extensions.extend(_str_list(meta.get("a2a_extensions")))
        deduped = tuple(dict.fromkeys(extensions))

        attempts = _as_int(
            a2a_meta.get("resubscribe_attempts")
            if "resubscribe_attempts" in a2a_meta
            else meta.get("a2a_resubscribe_attempts"),
            default=self._max_resubscribe_attempts,
        )
        configuration = self._build_message_configuration(meta, a2a_meta)
        return _SendOptions(
            configuration=configuration,
            extensions=deduped,
            max_resubscribe_attempts=max(attempts, 0),
        )

    def _build_message_configuration(
        self,
        root_meta: dict[str, Any],
        a2a_meta: dict[str, Any],
    ) -> MessageSendConfiguration | None:
        kwargs: dict[str, Any] = {}

        accepted_output_modes = _str_list(
            a2a_meta.get("accepted_output_modes")
            if "accepted_output_modes" in a2a_meta
            else root_meta.get("a2a_accepted_output_modes"),
        )
        if accepted_output_modes:
            kwargs["acceptedOutputModes"] = accepted_output_modes

        history_length = _as_int(
            a2a_meta.get("history_length")
            if "history_length" in a2a_meta
            else root_meta.get("a2a_history_length"),
            default=None,
        )
        if history_length is not None:
            kwargs["historyLength"] = max(history_length, 0)

        blocking = _as_bool(
            a2a_meta.get("blocking")
            if "blocking" in a2a_meta
            else root_meta.get("a2a_blocking"),
            default=None,
        )
        if blocking is not None:
            kwargs["blocking"] = blocking

        push_cfg_raw = (
            a2a_meta.get("push_notification")
            if "push_notification" in a2a_meta
            else root_meta.get("a2a_push_notification")
        )
        push_cfg = self._coerce_push_config(push_cfg_raw) or self._default_push_notification
        if push_cfg is not None:
            kwargs["pushNotificationConfig"] = push_cfg

        if not kwargs:
            return None
        return MessageSendConfiguration(**kwargs)

    def _coerce_push_config(
        self,
        value: PushNotificationConfig | dict[str, Any] | str | None,
    ) -> PushNotificationConfig | None:
        if value is None:
            return None
        if isinstance(value, PushNotificationConfig):
            return value
        if isinstance(value, str):
            return PushNotificationConfig(url=value)
        if not isinstance(value, dict):
            return None
        kwargs: dict[str, Any] = {}
        if value.get("url"):
            kwargs["url"] = str(value["url"])
        if value.get("token"):
            kwargs["token"] = str(value["token"])
        if value.get("id"):
            kwargs["id"] = str(value["id"])
        if "authentication" in value and value["authentication"] is not None:
            kwargs["authentication"] = value["authentication"]
        if not kwargs:
            return None
        return PushNotificationConfig(**kwargs)

    def _build_request(
        self,
        message: Any,
        metadata: dict[str, Any],
        options: _SendOptions,
    ) -> Any:
        payload = MessageSendParams(
            message=message,
            metadata=metadata,
            configuration=options.configuration,
        )
        return SendMessageRequest(id=str(uuid4()), params=payload)

    def _build_streaming_request(
        self,
        message: Any,
        metadata: dict[str, Any],
        options: _SendOptions,
    ) -> Any:
        payload = MessageSendParams(
            message=message,
            metadata=metadata,
            configuration=options.configuration,
        )
        return SendStreamingMessageRequest(id=str(uuid4()), params=payload)

    async def _iter_events(
        self,
        client: Any,
        message: Any,
        *,
        metadata: dict[str, Any],
        options: _SendOptions,
        streaming: bool,
    ) -> AsyncIterator[AgentEvent]:
        if streaming:
            async for event in self._iterate_with_resubscribe(
                client,
                message,
                metadata=metadata,
                options=options,
            ):
                yield event
            return

        iterator = await self._open_event_iterator(
            client,
            message,
            metadata=metadata,
            options=options,
            streaming=False,
        )
        async for raw_event in _iter_any(iterator):
            yield self._translate_event(raw_event)

    async def _iterate_with_resubscribe(
        self,
        client: Any,
        message: Any,
        *,
        metadata: dict[str, Any],
        options: _SendOptions,
    ) -> AsyncIterator[AgentEvent]:
        attempts = 0
        task_id: str | None = None
        use_resubscribe = False
        while True:
            iterator = await self._open_event_iterator(
                client,
                message,
                metadata=metadata,
                options=options,
                streaming=True,
                resubscribe_task_id=task_id if use_resubscribe else None,
            )
            try:
                async for raw_event in _iter_any(iterator):
                    event = self._translate_event(raw_event)
                    if event.task_id:
                        task_id = event.task_id
                    yield event
                return
            except Exception as exc:
                if (
                    task_id is None
                    or attempts >= options.max_resubscribe_attempts
                    or not hasattr(client, "resubscribe")
                    or not _is_stream_retryable(exc)
                ):
                    raise
                attempts += 1
                use_resubscribe = True
                logger.warning(
                    "A2A stream dropped for task %s; attempting resubscribe (%d/%d)",
                    task_id,
                    attempts,
                    options.max_resubscribe_attempts,
                )

    async def _open_event_iterator(
        self,
        client: Any,
        message: Any,
        *,
        metadata: dict[str, Any],
        options: _SendOptions,
        streaming: bool,
        resubscribe_task_id: str | None = None,
    ) -> Any:
        if resubscribe_task_id:
            return await self._call_resubscribe(
                client,
                task_id=resubscribe_task_id,
                metadata=metadata,
                extensions=options.extensions,
            )

        if streaming and hasattr(client, "send_message_streaming"):
            request = self._build_streaming_request(message, metadata, options)
            result = self._invoke_client(
                client.send_message_streaming,
                request,
                metadata=metadata,
                extensions=options.extensions,
            )
            return await _await_if_needed(result)

        if hasattr(client, "send_message"):
            return await self._call_send_message(
                client,
                message,
                metadata=metadata,
                options=options,
            )

        raise RuntimeError("A2A client does not expose send_message")

    async def _call_send_message(
        self,
        client: Any,
        message: Any,
        *,
        metadata: dict[str, Any],
        options: _SendOptions,
    ) -> Any:
        send = client.send_message
        result: Any

        # Modern A2A SDK API: send_message(Message, request_metadata=..., extensions=...)
        try:
            result = send(
                message,
                configuration=options.configuration,
                request_metadata=metadata or None,
                extensions=list(options.extensions) or None,
            )
            return await _await_if_needed(result)
        except TypeError:
            pass

        # Legacy API compatibility: send_message(SendMessageRequest)
        request = self._build_request(message, metadata, options)
        result = self._invoke_client(
            send,
            request,
            metadata=metadata,
            extensions=options.extensions,
        )
        return await _await_if_needed(result)

    async def _call_resubscribe(
        self,
        client: Any,
        *,
        task_id: str,
        metadata: dict[str, Any],
        extensions: tuple[str, ...],
    ) -> Any:
        params = TaskIdParams(id=task_id, metadata=metadata)
        result = self._invoke_client(
            client.resubscribe,
            params,
            metadata=metadata,
            extensions=extensions,
        )
        return await _await_if_needed(result)

    def _invoke_client(
        self,
        fn: Any,
        request: Any,
        *,
        metadata: dict[str, Any],
        extensions: tuple[str, ...],
        metadata_kwarg: str = "metadata",
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if extensions:
            kwargs["extensions"] = list(extensions)
        if metadata:
            kwargs[metadata_kwarg] = metadata
        try:
            return fn(request, **kwargs)
        except TypeError:
            # Retry with only extensions (older signatures vary).
            kwargs.pop(metadata_kwarg, None)
            if kwargs:
                try:
                    return fn(request, **kwargs)
                except TypeError:
                    pass
            return fn(request)

    def _translate_event(self, raw: Any) -> AgentEvent:
        """Convert a raw A2A SDK event/response into an ``AgentEvent``."""
        unwrapped = getattr(raw, "result", raw)

        # Handle (task, update) tuples
        if isinstance(unwrapped, tuple) and len(unwrapped) == 2:
            task, update = unwrapped
            return self._translate_task_event(task, update, raw)

        # Handle message objects with parts
        kind, text = self._extract_text(unwrapped)
        meta = self._extract_metadata(unwrapped)
        return AgentEvent(kind=kind, text=text, raw=raw, **meta)

    def _translate_task_event(self, task: Any, update: Any, raw: Any) -> AgentEvent:
        meta: dict[str, Any] = {}
        meta["task_id"] = getattr(task, "id", None) or getattr(task, "task_id", None)
        meta["context_id"] = getattr(task, "context_id", None)

        # State from task status
        status = getattr(task, "status", None)
        if status:
            state = getattr(status, "state", None)
            if state:
                meta["state"] = state.value if hasattr(state, "value") else str(state)
        if meta.get("state") in {"input-required", "auth-required"}:
            meta["requires_confirmation"] = True
        if meta.get("state") in {"rejected", "failed", "canceled", "completed"}:
            meta["is_final"] = True

        # Task metadata (branch info from ADK)
        task_meta = getattr(task, "metadata", None)
        if isinstance(task_meta, dict):
            meta["event_metadata"] = dict(task_meta)
            meta["branch"] = task_meta.get("branch")
            meta["author"] = task_meta.get("author")

        if update is None:
            # Try to get text from task message or result
            text = self._text_from_task(task)
            kind = "message" if text else "task"
            return AgentEvent(kind=kind, text=text, raw=raw, **meta)

        # Progress
        progress = getattr(update, "progress", None)
        if isinstance(progress, (int, float)):
            meta["progress"] = float(progress)

        meta["is_final"] = getattr(update, "final", False)

        # Artifact
        artifact = getattr(update, "artifact", None)
        if artifact:
            meta["artifact_id"] = getattr(artifact, "id", None)
            meta["artifact_name"] = getattr(artifact, "name", None)
            text = self._artifact_text(artifact)
            return AgentEvent(kind="artifact", text=text, raw=raw, **meta)

        # Status update
        status_val = getattr(update, "status", None)
        if status_val is not None:
            kind = "task" if meta.get("state") else "status"
            return AgentEvent(kind=kind, text=str(status_val), raw=raw, **meta)

        return AgentEvent(kind="event", text=None, raw=raw, **meta)

    def _text_from_task(self, task: Any) -> str | None:
        for attr in ("message", "result"):
            obj = getattr(task, attr, None)
            if obj is not None:
                text = self._message_text(obj)
                if text:
                    return text
        return None

    def _extract_text(self, obj: Any) -> tuple[str, str | None]:
        if hasattr(obj, "parts"):
            return "message", self._message_text(obj)
        status = getattr(obj, "status", None)
        if status is not None:
            return "status", str(status)
        return "event", None

    def _extract_metadata(self, obj: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        meta["task_id"] = getattr(obj, "task_id", None)
        meta["context_id"] = getattr(obj, "context_id", None)
        obj_meta = getattr(obj, "metadata", None)
        if isinstance(obj_meta, dict):
            meta["event_metadata"] = dict(obj_meta)
            meta["branch"] = obj_meta.get("branch")
            meta["author"] = obj_meta.get("author")
            if obj_meta.get("requires_confirmation"):
                meta["requires_confirmation"] = True
                meta["tool_call"] = obj_meta.get("tool_call")
        return meta

    def _message_text(self, message: Any) -> str | None:
        try:
            return get_message_text(message)
        except Exception:
            return self._parts_text(getattr(message, "parts", None))

    def _artifact_text(self, artifact: Any) -> str | None:
        try:
            return get_artifact_text(artifact)
        except Exception:
            return self._parts_text(getattr(artifact, "parts", None))

    def _parts_text(self, parts: Any) -> str | None:
        if not parts:
            return None
        texts: list[str] = []
        for part in parts:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and "text" in part:
                texts.append(str(part["text"]))
            else:
                t = getattr(part, "text", None)
                if isinstance(t, str):
                    texts.append(t)
        return "".join(texts) or None


def _is_stream_retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return any(hint in name or hint in message for hint in _STREAM_RETRY_HINTS)


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _iter_any(value: Any) -> AsyncIterator[Any]:
    if hasattr(value, "__aiter__"):
        async for item in value:
            yield item
        return
    yield value


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _as_int(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _as_bool(value: Any, *, default: bool | None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default
