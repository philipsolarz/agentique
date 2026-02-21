"""A2A agent adapter implementing the ``AgentAdapter`` protocol.

This is the primary adapter for communicating with agents that speak
the A2A protocol. It translates between agentique's core types and the
A2A SDK's message/request types.

Features:
    - Push notification configuration
    - Task resubscription for resilient streaming over Streamable HTTP
    - Extended agent card support (authenticated cards)
    - Auth-required / rejected state handling
    - A2A extension propagation via ``extensions`` parameter

Transport:
    The adapter communicates with A2A agents over Streamable HTTP
    (chunked transfer encoding). The A2A SDK client handles the
    underlying transport negotiation; this adapter does not contain
    any SSE-specific logic or EventSource retry headers.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, AsyncIterator
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


class A2AAgentAdapter:
    """Adapter that bridges agentique to A2A-protocol agents.

    Implements the ``AgentAdapter`` protocol via structural subtyping.
    Communicates with agents over Streamable HTTP (chunked transfer);
    there is no SSE-specific logic in this adapter.

    Args:
        agents: Mapping of agent name to AgentInfo.
        client_pool: Pool managing A2A SDK client instances.
        card_parser: Parser for extracting MCP components from agent cards.
        push_notification_url: Callback URL for push notification delivery.
        enable_reconnect: Automatically attempt task resubscription when the
            HTTP stream is interrupted before the task completes.
        max_reconnect_attempts: Maximum reconnection attempts before raising.
        extensions: A2A extension URIs to include in outgoing messages.
    """

    def __init__(
        self,
        agents: dict[str, AgentInfo],
        *,
        client_pool: A2AClientPool | None = None,
        card_parser: A2ACardParser | None = None,
        push_notification_url: str | None = None,
        enable_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
        extensions: list[str] | None = None,
    ) -> None:
        self._agents = dict(agents)
        self._pool = client_pool or A2AClientPool()
        self._parser = card_parser or A2ACardParser()
        self._push_url = push_notification_url
        self._enable_reconnect = enable_reconnect
        self._max_reconnect = max_reconnect_attempts
        self._extensions = extensions or []

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
        msg_obj, metadata = self._build_message(message, context)

        events: list[AgentEvent] = []
        async for event in self._iter_events(
            client, msg_obj, metadata=metadata, streaming=False,
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
        msg_obj, metadata = self._build_message(message, context)

        async for event in self._iter_events(
            client, msg_obj, metadata=metadata, streaming=True,
        ):
            yield event

    async def get_agent_card(
        self,
        agent_id: str,
        *,
        authenticated: bool = False,
    ) -> Any | None:
        """Fetch the agent card for *agent_id*.

        Args:
            agent_id: Agent identifier.
            authenticated: If True, attempt to fetch the extended
                (authenticated) agent card when the agent supports it.

        Returns:
            The raw agent card object, or None if not available.
        """
        info = self._require_agent(agent_id)
        client = await self._pool.get(info.base_url)

        # Try fetching the card
        getter = getattr(client, "get_card", None)
        if not callable(getter):
            return None

        result = getter()
        card = await result if inspect.isawaitable(result) else result

        if card is None:
            return None

        # Check if extended card is available and requested
        if authenticated:
            supports_extended = getattr(
                card, "supports_authenticated_extended_card", False,
            )
            if supports_extended:
                extended = await self._fetch_extended_card(client)
                if extended is not None:
                    return extended

        return card

    async def get_agent_extensions(
        self,
        agent_id: str,
    ) -> list[dict[str, Any]]:
        """Discover A2A extensions supported by an agent.

        Reads the agent card's ``capabilities.extensions`` field and
        returns a list of extension descriptors.

        Args:
            agent_id: Agent to query.

        Returns:
            List of extension dicts with ``uri``, ``description``,
            ``required``, and ``params`` keys.
        """
        card = await self.get_agent_card(agent_id)
        if card is None:
            return []

        capabilities = getattr(card, "capabilities", None)
        if capabilities is None:
            return []

        raw_extensions = getattr(capabilities, "extensions", None)
        if not raw_extensions:
            return []

        result: list[dict[str, Any]] = []
        for ext in raw_extensions:
            entry: dict[str, Any] = {"uri": getattr(ext, "uri", str(ext))}
            desc = getattr(ext, "description", None)
            if desc:
                entry["description"] = desc
            required = getattr(ext, "required", None)
            if required is not None:
                entry["required"] = required
            params = getattr(ext, "params", None)
            if params is not None:
                entry["params"] = params
            result.append(entry)
        return result

    async def cancel_task(self, agent_id: str, task_id: str) -> bool:
        """Send a ``tasks/cancel`` request to the A2A agent.

        Args:
            agent_id: The agent that owns the task.
            task_id: The task ID to cancel.

        Returns:
            ``True`` when the cancellation request was sent successfully.

        Raises:
            TaskNotCancelableError: When the A2A agent returns error -32004
                (task cannot be cancelled in its current state).
            UnsupportedOperationError: When the A2A client does not expose
                a cancellation method.
            AdapterError: On unexpected A2A communication failures.
        """
        from agentique.core.errors import (
            AdapterError,
            TaskNotCancelableError,
            UnsupportedOperationError,
        )

        info = self._require_agent(agent_id)
        client = await self._pool.get(info.base_url)

        cancel_fn = getattr(client, "cancel_task", None)
        if not callable(cancel_fn):
            raise UnsupportedOperationError(
                f"A2A client for agent '{agent_id}' does not support task cancellation"
            )

        try:
            params = TaskIdParams(id=task_id)
            result = cancel_fn(params)
            if inspect.isawaitable(result):
                result = await result
            logger.info("Cancel request sent for task %s on agent %s", task_id, agent_id)
            return True

        except Exception as exc:
            # Map A2A error code -32004 to TaskNotCancelableError
            code = getattr(exc, "code", None) or getattr(exc, "error_code", None)
            msg = str(exc)
            if code == -32004 or "TaskNotCancelable" in type(exc).__name__ or "-32004" in msg:
                raise TaskNotCancelableError(
                    f"Task '{task_id}' cannot be cancelled: {exc}"
                ) from exc
            raise AdapterError(
                f"Failed to cancel task '{task_id}' on agent '{agent_id}': {exc}"
            ) from exc

    async def close(self) -> None:
        await self._pool.close()

    # ---- Push notification support ----

    async def configure_push_notifications(
        self,
        agent_id: str,
        task_id: str,
        callback_url: str | None = None,
    ) -> bool:
        """Configure push notifications for a task on the A2A agent.

        Args:
            agent_id: Target agent.
            task_id: Task to configure notifications for.
            callback_url: URL where the agent should POST task updates.
                Falls back to the adapter-level ``push_notification_url``.

        Returns:
            True if push notifications were configured successfully.
        """
        url = callback_url or self._push_url
        if not url:
            logger.debug("No push notification URL configured")
            return False

        info = self._require_agent(agent_id)
        client = await self._pool.get(info.base_url)

        try:
            push_config = PushNotificationConfig(
                url=url,
                id=str(uuid4()),
                token=str(uuid4()),
            )

            # Use the client's push notification API if available
            set_push = getattr(client, "set_push_notification_config", None)
            if callable(set_push):
                result = set_push(task_id, push_config)
                if inspect.isawaitable(result):
                    await result
                logger.info(
                    "Push notifications configured for task %s on agent %s",
                    task_id, agent_id,
                )
                return True

            logger.debug(
                "Agent '%s' client does not support push notification config",
                agent_id,
            )
        except Exception:
            logger.warning(
                "Failed to configure push notifications for task %s",
                task_id, exc_info=True,
            )

        return False

    # ---- Task resubscription ----

    async def resubscribe(
        self,
        agent_id: str,
        task_id: str,
    ) -> AsyncIterator[AgentEvent]:
        """Resubscribe to a task's event stream after disconnection.

        Uses A2A's ``tasks/resubscribe`` method for resilient streaming.

        Args:
            agent_id: Agent that owns the task.
            task_id: Task ID to resubscribe to.

        Yields:
            AgentEvent instances from the resumed stream.
        """
        info = self._require_agent(agent_id)
        client = await self._pool.get(info.base_url)

        resubscribe_fn = getattr(client, "resubscribe", None)
        if not callable(resubscribe_fn):
            logger.warning(
                "Client for agent '%s' does not support resubscription",
                agent_id,
            )
            return

        try:
            params = TaskIdParams(id=task_id)
            result = resubscribe_fn(params)
            if inspect.isawaitable(result):
                result = await result

            if hasattr(result, "__aiter__"):
                async for raw_event in result:
                    yield self._translate_event(raw_event)
            else:
                yield self._translate_event(result)

        except Exception:
            logger.warning(
                "Resubscription failed for task %s on agent %s",
                task_id, agent_id, exc_info=True,
            )

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
    ) -> tuple[Any, dict[str, Any]]:
        msg = create_text_message_object(content=text)
        metadata: dict[str, Any] = {"mcp": context.to_metadata()}
        if context.conversation_history:
            metadata["conversation_history"] = context.conversation_history

        # Inject W3C trace context for end-to-end distributed tracing
        from agentique.extensions import (
            TRACE_CONTEXT_URI,
            current_trace_context,
            pack_trace_context,
        )
        trace_ctx = current_trace_context()
        if trace_ctx:
            metadata[TRACE_CONTEXT_URI] = pack_trace_context(**trace_ctx)

        # Attach extensions to the message
        update_fields: dict[str, Any] = {"metadata": metadata}
        if self._extensions:
            update_fields["extensions"] = list(self._extensions)

        if hasattr(msg, "model_copy"):
            msg = msg.model_copy(update=update_fields)
        else:
            if hasattr(msg, "metadata"):
                try:
                    msg.metadata = metadata
                except Exception:
                    pass
            if self._extensions and hasattr(msg, "extensions"):
                try:
                    msg.extensions = list(self._extensions)
                except Exception:
                    pass

        return msg, metadata

    def _build_request(self, message: Any, metadata: dict[str, Any]) -> Any:
        configuration = MessageSendConfiguration() if MessageSendConfiguration else None
        payload = MessageSendParams(
            message=message, metadata=metadata, configuration=configuration,
        )
        return SendMessageRequest(id=str(uuid4()), params=payload)

    def _build_streaming_request(self, message: Any, metadata: dict[str, Any]) -> Any:
        configuration = MessageSendConfiguration() if MessageSendConfiguration else None
        payload = MessageSendParams(
            message=message, metadata=metadata, configuration=configuration,
        )
        return SendStreamingMessageRequest(id=str(uuid4()), params=payload)

    async def _fetch_extended_card(self, client: Any) -> Any | None:
        """Attempt to fetch the authenticated extended agent card."""
        try:
            # Check for extended card retrieval methods
            get_extended = getattr(
                client, "get_authenticated_extended_card", None,
            )
            if callable(get_extended):
                result = get_extended()
                return await result if inspect.isawaitable(result) else result

            # Some SDK versions use get_card with auth param
            get_card = getattr(client, "get_card", None)
            if callable(get_card):
                sig = inspect.signature(get_card)
                if "authenticated" in sig.parameters or "extended" in sig.parameters:
                    result = get_card(authenticated=True)
                    return await result if inspect.isawaitable(result) else result

        except Exception:
            logger.debug(
                "Failed to fetch extended agent card", exc_info=True,
            )
        return None

    async def _iter_events(
        self,
        client: Any,
        message: Any,
        *,
        metadata: dict[str, Any],
        streaming: bool,
    ) -> AsyncIterator[AgentEvent]:
        """Iterate events from the A2A agent over Streamable HTTP.

        Consumes the chunked response stream produced by the A2A SDK client.
        On transient stream disruptions (network errors, timeouts) the method
        attempts task resubscription up to ``max_reconnect_attempts`` times
        before propagating the error.
        """
        last_task_id: str | None = None

        if not hasattr(client, "send_message"):
            raise RuntimeError("A2A client does not expose send_message")

        attempt = 0

        while True:
            try:
                result = await self._call_send_message(
                    client, message, metadata=metadata, streaming=streaming,
                )

                if hasattr(result, "__aiter__"):
                    async for raw_event in result:
                        event = self._translate_event(raw_event)
                        if event.task_id:
                            last_task_id = event.task_id
                        yield event
                else:
                    event = self._translate_event(result)
                    if event.task_id:
                        last_task_id = event.task_id
                    yield event
                return  # Stream completed normally

            except (ConnectionError, OSError, asyncio.TimeoutError, EOFError) as exc:
                # Transient stream disruption — attempt resubscription
                attempt += 1
                if (
                    not self._enable_reconnect
                    or attempt > self._max_reconnect
                    or not last_task_id
                ):
                    raise

                logger.warning(
                    "HTTP stream interrupted (attempt %d/%d), "
                    "resubscribing to task %s",
                    attempt, self._max_reconnect, last_task_id,
                )

                resubscribe_fn = getattr(client, "resubscribe", None)
                if not callable(resubscribe_fn):
                    raise

                try:
                    params = TaskIdParams(id=last_task_id)
                    result = resubscribe_fn(params)
                    if inspect.isawaitable(result):
                        result = await result

                    if hasattr(result, "__aiter__"):
                        async for raw_event in result:
                            event = self._translate_event(raw_event)
                            yield event
                        return  # Resubscription completed successfully
                    else:
                        yield self._translate_event(result)
                        return
                except Exception as resub_exc:
                    logger.warning("Resubscription failed: %s", resub_exc)
                    if attempt >= self._max_reconnect:
                        raise exc from resub_exc
                    continue

    async def _call_send_message(
        self,
        client: Any,
        message: Any,
        *,
        metadata: dict[str, Any],
        streaming: bool,
    ) -> Any:
        """Call the client's send_message with the appropriate API style.

        a2a-sdk 0.3.22 BaseClient.send_message() accepts a Message directly
        with kwargs (configuration, request_metadata). Older or mock clients
        may accept a SendMessageRequest/MessageSendParams positionally.
        """
        # Try a2a-sdk 0.3.22+ BaseClient API first: send_message(Message, **kwargs)
        try:
            configuration = MessageSendConfiguration() if MessageSendConfiguration else None
            result = client.send_message(
                message,
                configuration=configuration,
                request_metadata=metadata,
            )
            if inspect.isawaitable(result):
                result = await result
            return result
        except TypeError:
            pass  # Fall back to legacy API

        # Legacy: try send_message_streaming with wrapped request
        if hasattr(client, "send_message_streaming") and streaming:
            request = self._build_streaming_request(message, metadata)
            result = client.send_message_streaming(request)
            if inspect.isawaitable(result):
                result = await result
            return result

        # Legacy: send_message with wrapped request
        request = self._build_request(message, metadata)
        result = client.send_message(request)
        if inspect.isawaitable(result):
            result = await result
        return result

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
            return AgentEvent(kind="status", text=str(status_val), raw=raw, **meta)

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
        # Capture A2A extension URIs from the response
        extensions = getattr(obj, "extensions", None)
        if extensions and isinstance(extensions, (list, tuple)):
            meta["extensions"] = list(extensions)
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
