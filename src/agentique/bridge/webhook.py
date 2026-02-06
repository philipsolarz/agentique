"""Webhook receiver for agent push notifications.

Accepts push notifications from A2A agents and translates them into
internal events (task status updates, messages, artifacts) that flow
through agentique's event system.

The receiver exposes itself as a FastMCP resource so that MCP clients
can subscribe to notification feeds, and also emits events via the
``AsyncEventEmitter`` for programmatic consumption.

Usage::

    from agentique.bridge.webhook import WebhookReceiver

    receiver = WebhookReceiver(emitter=emitter, task_manager=tasks)
    receiver.register(mcp)  # adds resource + tool to the FastMCP server
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agentique.core.events import AsyncEventEmitter
from agentique.core.types import AgentEvent, TaskState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PushNotification:
    """Parsed push notification from an agent."""

    id: str
    agent_id: str
    task_id: str | None = None
    kind: str = "status"  # status, message, artifact, error
    state: str | None = None
    text: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "state": self.state,
            "text": self.text,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class WebhookReceiver:
    """Receives push notifications from agents and translates them.

    Maintains an ordered log of received notifications and emits
    events through the ``AsyncEventEmitter`` for each notification.

    Events emitted:
        - ``webhook.received`` — raw notification data
        - ``task.state_changed`` — when notification includes state change
        - ``message.received`` — when notification includes a message

    Args:
        emitter: Event emitter for dispatching notification events.
        task_manager: Optional task manager for automatic state updates.
        max_history: Maximum number of notifications to keep in memory.
    """

    def __init__(
        self,
        emitter: AsyncEventEmitter,
        task_manager: Any | None = None,
        max_history: int = 1000,
    ) -> None:
        self._emitter = emitter
        self._task_manager = task_manager
        self._max_history = max_history
        self._notifications: list[PushNotification] = []
        self._subscriptions: dict[str, list[str]] = {}  # agent_id -> [task_ids]

    async def receive(self, payload: dict[str, Any]) -> PushNotification:
        """Process an incoming push notification.

        Parses the payload, stores it, updates task state if applicable,
        and emits events.

        Args:
            payload: Raw notification payload (JSON-decoded dict).

        Returns:
            The parsed ``PushNotification``.
        """
        notification = _parse_notification(payload)

        # Store notification
        self._notifications.append(notification)
        if len(self._notifications) > self._max_history:
            self._notifications = self._notifications[-self._max_history:]

        # Emit raw event
        await self._emitter.emit(
            "webhook.received",
            notification=notification,
            payload=payload,
        )

        # Update task state if we have a task manager and task_id
        if self._task_manager and notification.task_id and notification.state:
            try:
                tracker = await self._task_manager.get_or_none(notification.task_id)
                if tracker:
                    try:
                        new_state = TaskState(notification.state)
                        tracker.transition(new_state, notification.text)
                        await self._emitter.emit(
                            "task.state_changed",
                            task_id=notification.task_id,
                            state=notification.state,
                        )
                    except ValueError:
                        logger.debug(
                            "Unknown task state in notification: %s",
                            notification.state,
                        )
            except Exception:
                logger.debug(
                    "Failed to update task from notification",
                    exc_info=True,
                )

        # Emit typed events
        if notification.kind == "message" and notification.text:
            await self._emitter.emit(
                "message.received",
                agent_id=notification.agent_id,
                text=notification.text,
                task_id=notification.task_id,
            )

        return notification

    def subscribe(self, agent_id: str, task_id: str) -> None:
        """Register interest in notifications for a specific agent/task.

        Args:
            agent_id: The agent to subscribe to.
            task_id: The task to subscribe to.
        """
        if agent_id not in self._subscriptions:
            self._subscriptions[agent_id] = []
        if task_id not in self._subscriptions[agent_id]:
            self._subscriptions[agent_id].append(task_id)

    def unsubscribe(self, agent_id: str, task_id: str | None = None) -> None:
        """Remove subscription for an agent/task.

        Args:
            agent_id: The agent to unsubscribe from.
            task_id: Specific task, or None to remove all for this agent.
        """
        if task_id is None:
            self._subscriptions.pop(agent_id, None)
        elif agent_id in self._subscriptions:
            try:
                self._subscriptions[agent_id].remove(task_id)
            except ValueError:
                pass

    def get_notifications(
        self,
        agent_id: str | None = None,
        task_id: str | None = None,
        limit: int = 50,
    ) -> list[PushNotification]:
        """Retrieve stored notifications with optional filtering.

        Args:
            agent_id: Filter by agent.
            task_id: Filter by task.
            limit: Maximum number to return.

        Returns:
            List of matching notifications (most recent first).
        """
        results = self._notifications
        if agent_id:
            results = [n for n in results if n.agent_id == agent_id]
        if task_id:
            results = [n for n in results if n.task_id == task_id]
        return list(reversed(results[-limit:]))

    def register(self, mcp: Any) -> None:
        """Register webhook tools and resources on a FastMCP server.

        Adds:
            - ``webhook_notify`` tool: accepts push notifications
            - ``webhook://notifications`` resource: lists recent notifications

        Args:
            mcp: The FastMCP server instance.
        """
        receiver = self

        @mcp.tool(name="webhook_notify")
        async def webhook_notify_tool(payload: str) -> str:
            """Receive a push notification from an agent.

            Args:
                payload: JSON-encoded notification payload
            """
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON payload"})

            notification = await receiver.receive(data)
            return json.dumps({
                "status": "received",
                "notification_id": notification.id,
            })

        @mcp.resource("webhook://notifications")
        def webhook_notifications_resource() -> str:
            """Recent push notifications received by the webhook."""
            notifications = receiver.get_notifications(limit=50)
            return json.dumps(
                {"notifications": [n.to_dict() for n in notifications]},
                indent=2,
            )

    @property
    def notification_count(self) -> int:
        """Total number of stored notifications."""
        return len(self._notifications)

    def clear(self) -> None:
        """Clear all stored notifications and subscriptions."""
        self._notifications.clear()
        self._subscriptions.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_notification(payload: dict[str, Any]) -> PushNotification:
    """Parse a raw notification payload into a ``PushNotification``."""
    # Support both A2A-style and generic payloads
    notification_id = str(payload.get("id", uuid4()))

    # A2A push notifications use nested "result" structure
    task_state = payload.get("result")
    if isinstance(task_state, dict) and task_state:
        agent_id = task_state.get("agent_id", payload.get("agent_id", "unknown"))
        task_id = task_state.get("id", payload.get("task_id"))
        status = task_state.get("status", {})
        if isinstance(status, dict):
            state = status.get("state", payload.get("state"))
            message = status.get("message", {})
            text = None
            if isinstance(message, dict):
                parts = message.get("parts", [])
                texts = []
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        texts.append(part["text"])
                text = " ".join(texts) if texts else None
            elif isinstance(message, str):
                text = message
        else:
            state = payload.get("state")
            text = payload.get("text")
    else:
        # Simple/generic payload format
        agent_id = payload.get("agent_id", "unknown")
        task_id = payload.get("task_id")
        state = payload.get("state")
        text = payload.get("text")

    kind = payload.get("kind", "status")
    if text and kind == "status":
        kind = "message"

    timestamp = payload.get("timestamp", datetime.now(timezone.utc).isoformat())

    return PushNotification(
        id=notification_id,
        agent_id=agent_id,
        task_id=task_id,
        kind=kind,
        state=state,
        text=text,
        data=payload,
        timestamp=timestamp,
    )
