"""Tests for the webhook receiver module."""

from __future__ import annotations

import json
import pytest

from agentique.core.events import AsyncEventEmitter
from agentique.core.types import TaskState, TaskTracker
from agentique.bridge.webhook import PushNotification, WebhookReceiver, _parse_notification


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def emitter():
    return AsyncEventEmitter()


class FakeTaskManager:
    """Minimal task manager stub for testing."""

    def __init__(self):
        self.tasks: dict[str, TaskTracker] = {}

    async def create(self, task_id: str, context_id: str | None = None) -> TaskTracker:
        tracker = TaskTracker(task_id=task_id, context_id=context_id)
        self.tasks[task_id] = tracker
        return tracker

    async def get_or_none(self, task_id: str) -> TaskTracker | None:
        return self.tasks.get(task_id)


@pytest.fixture
def task_manager():
    return FakeTaskManager()


@pytest.fixture
def receiver(emitter, task_manager):
    return WebhookReceiver(emitter=emitter, task_manager=task_manager)


# ---------------------------------------------------------------------------
# Notification parsing tests
# ---------------------------------------------------------------------------


class TestParseNotification:
    def test_simple_payload(self):
        payload = {
            "agent_id": "test-agent",
            "task_id": "task-1",
            "state": "completed",
            "text": "Done",
            "kind": "message",
        }
        n = _parse_notification(payload)
        assert n.agent_id == "test-agent"
        assert n.task_id == "task-1"
        assert n.state == "completed"
        assert n.text == "Done"
        assert n.kind == "message"

    def test_a2a_style_payload(self):
        """A2A push notifications have a nested 'result' structure."""
        payload = {
            "id": "notif-123",
            "result": {
                "id": "task-abc",
                "agent_id": "a2a-agent",
                "status": {
                    "state": "working",
                    "message": {"parts": [{"text": "Processing..."}]},
                },
            },
        }
        n = _parse_notification(payload)
        assert n.id == "notif-123"
        assert n.agent_id == "a2a-agent"
        assert n.task_id == "task-abc"
        assert n.state == "working"
        assert n.text == "Processing..."

    def test_payload_without_agent_id(self):
        payload = {"state": "completed"}
        n = _parse_notification(payload)
        assert n.agent_id == "unknown"

    def test_payload_generates_id(self):
        payload = {"agent_id": "test"}
        n = _parse_notification(payload)
        assert n.id  # Should have a generated UUID

    def test_notification_to_dict(self):
        n = PushNotification(
            id="n1", agent_id="agent-a", task_id="t1",
            kind="status", state="working", text="busy",
        )
        d = n.to_dict()
        assert d["id"] == "n1"
        assert d["agent_id"] == "agent-a"
        assert d["kind"] == "status"


# ---------------------------------------------------------------------------
# WebhookReceiver tests
# ---------------------------------------------------------------------------


class TestWebhookReceiver:
    @pytest.mark.anyio
    async def test_receive_stores_notification(self, receiver):
        payload = {"agent_id": "test", "kind": "status"}
        await receiver.receive(payload)
        assert receiver.notification_count == 1

    @pytest.mark.anyio
    async def test_receive_emits_event(self, receiver, emitter):
        events = []
        emitter.on("webhook.received", lambda **kw: events.append(kw))

        await receiver.receive({"agent_id": "test"})
        assert len(events) == 1
        assert events[0]["notification"].agent_id == "test"

    @pytest.mark.anyio
    async def test_receive_updates_task_state(self, receiver, task_manager):
        tracker = await task_manager.create("task-1")
        assert tracker.state == TaskState.submitted

        await receiver.receive({
            "agent_id": "test",
            "task_id": "task-1",
            "state": "completed",
        })
        assert tracker.state == TaskState.completed

    @pytest.mark.anyio
    async def test_receive_message_emits_message_event(self, receiver, emitter):
        events = []
        emitter.on("message.received", lambda **kw: events.append(kw))

        await receiver.receive({
            "agent_id": "test",
            "kind": "message",
            "text": "Hello",
        })
        assert len(events) == 1
        assert events[0]["text"] == "Hello"

    @pytest.mark.anyio
    async def test_max_history(self, emitter):
        receiver = WebhookReceiver(emitter=emitter, max_history=5)
        for i in range(10):
            await receiver.receive({"agent_id": f"agent-{i}"})
        assert receiver.notification_count == 5

    @pytest.mark.anyio
    async def test_get_notifications_filters(self, receiver):
        await receiver.receive({"agent_id": "agent-a", "task_id": "t1"})
        await receiver.receive({"agent_id": "agent-b", "task_id": "t2"})
        await receiver.receive({"agent_id": "agent-a", "task_id": "t3"})

        by_agent = receiver.get_notifications(agent_id="agent-a")
        assert len(by_agent) == 2

        by_task = receiver.get_notifications(task_id="t2")
        assert len(by_task) == 1

    def test_subscribe_unsubscribe(self, receiver):
        receiver.subscribe("agent-a", "task-1")
        assert "agent-a" in receiver._subscriptions
        assert "task-1" in receiver._subscriptions["agent-a"]

        receiver.unsubscribe("agent-a", "task-1")
        assert "task-1" not in receiver._subscriptions["agent-a"]

        receiver.subscribe("agent-b", "task-2")
        receiver.unsubscribe("agent-b")
        assert "agent-b" not in receiver._subscriptions

    def test_clear(self, receiver):
        receiver.subscribe("a", "t1")
        receiver.clear()
        assert receiver.notification_count == 0
        assert len(receiver._subscriptions) == 0
