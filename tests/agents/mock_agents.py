"""Mock A2A agents for deterministic testing.

These agents implement the a2a-sdk's AgentExecutor interface and provide
predictable behaviors for testing different protocol features.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from a2a.types import Artifact, TaskState, TaskStatus, TextPart
from a2a.utils import new_agent_text_message

if TYPE_CHECKING:
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import RequestContext


class EchoAgent:
    """Returns the input message verbatim. Tests basic routing."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Echo back the user's message."""
        user_text = ""
        if context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text"):
                    user_text += part.text

        await event_queue.enqueue_event(new_agent_text_message(f"echo: {user_text}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation (no-op for echo)."""
        pass


class StreamingAgent:
    """Sends 5 artifact chunks then completes. Tests streaming fidelity."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Send multiple chunks to test streaming."""
        from a2a.types import TaskArtifactUpdateEvent

        for i in range(5):
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    taskId=context.task_id,
                    contextId=context.context_id,
                    artifact=Artifact(
                        artifactId=f"chunk-{i}",
                        parts=[TextPart(text=f"chunk {i} of 4")],
                    ),
                    append=True,
                    lastChunk=(i == 4),
                )
            )
            await asyncio.sleep(0.01)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation."""
        pass


class ErrorAgent:
    """Immediately fails with a known error code. Tests error mapping."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Fail immediately with an error."""
        from a2a.types import TaskStatusUpdateEvent

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=context.task_id,
                contextId=context.context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message("Intentional failure for testing"),
                ),
                final=True,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation."""
        pass


class InputRequiredAgent:
    """Transitions to input-required state. Tests elicitation flow."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Request additional input from the user."""
        from a2a.types import TaskStatusUpdateEvent

        # Check if this is a follow-up with the required input
        user_text = ""
        if context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text"):
                    user_text += part.text

        if user_text.startswith("CONFIRM:"):
            await event_queue.enqueue_event(new_agent_text_message(f"Confirmed: {user_text[8:]}"))
        else:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    taskId=context.task_id,
                    contextId=context.context_id,
                    status=TaskStatus(
                        state=TaskState.input_required,
                        message=new_agent_text_message("Please confirm by replying CONFIRM:<your input>"),
                    ),
                    final=False,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation."""
        pass


class LongRunningAgent:
    """Sends progress updates over 2 seconds. Tests task polling."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Simulate a long-running task with progress updates."""
        from a2a.types import TaskStatusUpdateEvent

        for step in range(10):
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    taskId=context.task_id,
                    contextId=context.context_id,
                    status=TaskStatus(state=TaskState.working),
                    final=False,
                )
            )
            await asyncio.sleep(0.2)

        await event_queue.enqueue_event(new_agent_text_message("Long task completed"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation by transitioning to canceled state."""
        from a2a.types import TaskStatusUpdateEvent

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=context.task_id,
                contextId=context.context_id,
                status=TaskStatus(state=TaskState.canceled),
                final=True,
            )
        )


class CalculatorAgent:
    """Simple calculator for testing basic operations."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Parse and evaluate simple math expressions."""
        user_text = ""
        if context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text"):
                    user_text += part.text

        try:
            # Very simple calculator - just look for patterns like "2 + 3" or "10 * 5"
            import re

            # Match patterns like "what is 2 + 3", "calculate 10 * 5", etc.
            match = re.search(r"(\d+)\s*([+\-*/])\s*(\d+)", user_text)
            if match:
                a, op, b = match.groups()
                a, b = int(a), int(b)
                if op == "+":
                    result = a + b
                elif op == "-":
                    result = a - b
                elif op == "*":
                    result = a * b
                elif op == "/":
                    result = a / b if b != 0 else "Error: Division by zero"
                else:
                    result = "Unknown operation"

                await event_queue.enqueue_event(new_agent_text_message(f"The answer is {result}"))
            else:
                await event_queue.enqueue_event(
                    new_agent_text_message("I can calculate simple expressions like '2 + 3' or '10 * 5'")
                )
        except Exception as e:
            await event_queue.enqueue_event(new_agent_text_message(f"Error: {e}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation."""
        pass
