"""Integration tests for push notification / webhook functionality.

Tests webhook delivery using the WebhookCollector fixture.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.integration
async def test_webhook_collector_receives_notification(webhook_server):
    """WebhookCollector can receive and record webhook POSTs."""
    import httpx

    async with httpx.AsyncClient() as client:
        # Send a test webhook
        payload = {"task_id": "test-123", "status": "completed"}
        response = await client.post(webhook_server["url"], json=payload)
        assert response.status_code == 200

    # Verify the collector received it
    collector = webhook_server["collector"]
    assert len(collector.notifications) == 1
    assert collector.notifications[0]["task_id"] == "test-123"


@pytest.mark.integration
async def test_webhook_collector_wait_for_notification(webhook_server):
    """WebhookCollector.wait_for_notification() returns when data arrives."""
    import httpx

    collector = webhook_server["collector"]

    async def send_webhook():
        await asyncio.sleep(0.1)  # Small delay
        async with httpx.AsyncClient() as client:
            await client.post(webhook_server["url"], json={"test": "data"})

    # Start sending in background
    task = asyncio.create_task(send_webhook())

    # Wait for notification
    notification = await collector.wait_for_notification(timeout=2.0)
    assert notification["test"] == "data"

    await task


@pytest.mark.integration
async def test_webhook_collector_timeout(webhook_server):
    """WebhookCollector.wait_for_notification() raises on timeout."""
    collector = webhook_server["collector"]

    with pytest.raises(AssertionError, match="No webhook received"):
        await collector.wait_for_notification(timeout=0.5)


@pytest.mark.integration
async def test_webhook_collector_multiple_notifications(webhook_server):
    """WebhookCollector tracks multiple notifications."""
    import httpx

    async with httpx.AsyncClient() as client:
        for i in range(3):
            await client.post(webhook_server["url"], json={"count": i})

    collector = webhook_server["collector"]
    assert len(collector.notifications) == 3
    assert collector.notifications[0]["count"] == 0
    assert collector.notifications[1]["count"] == 1
    assert collector.notifications[2]["count"] == 2


@pytest.mark.integration
async def test_webhook_collector_clear(webhook_server):
    """WebhookCollector.clear() resets the notification list."""
    import httpx

    collector = webhook_server["collector"]

    async with httpx.AsyncClient() as client:
        await client.post(webhook_server["url"], json={"test": 1})

    assert len(collector.notifications) == 1

    collector.clear()
    assert len(collector.notifications) == 0


@pytest.mark.integration
async def test_webhook_server_returns_ok(webhook_server):
    """Webhook server returns {status: ok} for successful POSTs."""
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(webhook_server["url"], json={"data": "test"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
