"""A2A agent adapter implementing the ``AgentAdapter`` protocol.

This is the primary adapter for communicating with agents that speak
the A2A protocol. It translates between agentique's core types and the
A2A SDK's message/request types.
"""

from __future__ import annotations

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
        SendMessageRequest,
        SendStreamingMessageRequest,
    )
except ImportError as exc:
    raise RuntimeError(
        "a2a-sdk is required for the A2A adapter. "
        "Install with: pip install 'agentique[a2a]'"
    ) from exc


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
    ) -> None:
        self._agents = dict(agents)
        self._pool = client_pool or A2AClientPool()
        self._parser = card_parser or A2ACardParser()

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

    async def get_agent_card(self, agent_id: str) -> Any | None:
        """Fetch the raw agent card for *agent_id*."""
        info = self._require_agent(agent_id)
        client = await self._pool.get(info.base_url)
        getter = getattr(client, "get_card", None)
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
    ) -> tuple[Any, dict[str, Any]]:
        msg = create_text_message_object(content=text)
        metadata: dict[str, Any] = {"mcp": context.to_metadata()}
        if context.conversation_history:
            metadata["conversation_history"] = context.conversation_history
        # Attach metadata to message
        if hasattr(msg, "model_copy"):
            msg = msg.model_copy(update={"metadata": metadata})
        elif hasattr(msg, "metadata"):
            try:
                msg.metadata = metadata
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

    async def _iter_events(
        self,
        client: Any,
        message: Any,
        *,
        metadata: dict[str, Any],
        streaming: bool,
    ) -> AsyncIterator[AgentEvent]:
        if hasattr(client, "send_message_streaming") and streaming:
            request = self._build_streaming_request(message, metadata)
            iterator = client.send_message_streaming(request)
            async for raw_event in iterator:
                yield self._translate_event(raw_event)
            return

        # Non-streaming fallback
        request = self._build_request(message, metadata)
        if hasattr(client, "send_message"):
            response = await client.send_message(request)
            yield self._translate_event(response)
            return

        raise RuntimeError("A2A client does not expose send_message")

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
