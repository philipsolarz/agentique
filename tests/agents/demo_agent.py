"""Advanced demo agent for interactive testing.

This agent demonstrates all A2A protocol features and is designed for
manual testing with real MCP clients (Claude CLI, GitHub Copilot, etc.).

Features demonstrated:
- Multi-turn conversations with context/memory
- Streaming responses
- Error handling and recovery
- Input elicitation (input_required state)
- Background tasks with progress updates
- All message types and artifact handling
- Realistic response generation
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from a2a.types import Artifact, TaskState, TaskStatus, TextPart
from a2a.utils import new_agent_text_message

if TYPE_CHECKING:
    from a2a.server.agent_execution import AgentExecutor
    from a2a.server.events import EventQueue
    from a2a.server.request_handlers import RequestContext


class ConversationMemory:
    """Simple conversation memory for tracking context across turns."""

    def __init__(self):
        self.conversations: dict[str, list[dict[str, Any]]] = {}

    def add_message(self, context_id: str, role: str, text: str):
        """Add a message to conversation history."""
        if context_id not in self.conversations:
            self.conversations[context_id] = []

        self.conversations[context_id].append(
            {"role": role, "text": text, "timestamp": datetime.now().isoformat()}
        )

    def get_history(self, context_id: str) -> list[dict[str, Any]]:
        """Get conversation history for a context."""
        return self.conversations.get(context_id, [])

    def get_summary(self, context_id: str) -> str:
        """Get a text summary of the conversation."""
        history = self.get_history(context_id)
        if not history:
            return "No conversation history."

        lines = []
        for msg in history[-5:]:  # Last 5 messages
            lines.append(f"{msg['role']}: {msg['text'][:100]}")
        return "\n".join(lines)


class DemoAgent:
    """Comprehensive demo agent showcasing all A2A/MCP bridge features.

    Command syntax (case-insensitive):
    - /help - Show all available commands
    - /echo <text> - Echo back the text
    - /stream <text> - Stream the response in chunks
    - /error - Trigger an error
    - /ask - Request additional input (input_required state)
    - /background <task> - Run a background task with progress
    - /calc <expression> - Calculate a math expression
    - /memory - Show conversation history
    - /context - Show current context information
    - /slow <text> - Respond slowly to test timeouts
    - /multipart - Return a multipart response
    - Normal conversation - Engage in multi-turn conversation
    """

    def __init__(self):
        self.memory = ConversationMemory()
        self._background_tasks: dict[str, dict] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute agent logic based on user message."""
        # Extract user text
        user_text = self._extract_text(context)

        # Store in memory
        context_id = context.context_id or "default"
        self.memory.add_message(context_id, "user", user_text)

        # Parse command
        command = self._parse_command(user_text)

        # Route to appropriate handler
        if command == "/help":
            await self._handle_help(event_queue)
        elif command == "/echo":
            await self._handle_echo(user_text, event_queue)
        elif command == "/stream":
            await self._handle_stream(user_text, event_queue)
        elif command == "/error":
            await self._handle_error(context, event_queue)
        elif command == "/ask":
            await self._handle_ask(context, event_queue)
        elif command == "/background":
            await self._handle_background(user_text, context, event_queue)
        elif command == "/calc":
            await self._handle_calc(user_text, event_queue)
        elif command == "/memory":
            await self._handle_memory(context_id, event_queue)
        elif command == "/context":
            await self._handle_context(context, event_queue)
        elif command == "/slow":
            await self._handle_slow(user_text, event_queue)
        elif command == "/multipart":
            await self._handle_multipart(event_queue)
        else:
            await self._handle_conversation(user_text, context_id, event_queue)

        # Store assistant response in memory
        # (In real implementation, would extract from event_queue)
        self.memory.add_message(context_id, "assistant", f"Responded to: {command or 'conversation'}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation."""
        from a2a.types import TaskStatusUpdateEvent

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=context.task_id,
                contextId=context.context_id,
                status=TaskStatus(
                    state=TaskState.canceled,
                    message=new_agent_text_message("Task cancelled by user"),
                ),
                final=True,
            )
        )

    def _extract_text(self, context: RequestContext) -> str:
        """Extract text from message parts."""
        text = ""
        if context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text"):
                    text += part.text
        return text.strip()

    def _parse_command(self, text: str) -> str | None:
        """Parse command from user text."""
        text_lower = text.lower().strip()
        commands = [
            "/help",
            "/echo",
            "/stream",
            "/error",
            "/ask",
            "/background",
            "/calc",
            "/memory",
            "/context",
            "/slow",
            "/multipart",
        ]

        for cmd in commands:
            if text_lower.startswith(cmd):
                return cmd
        return None

    async def _handle_help(self, event_queue: EventQueue) -> None:
        """Show help message."""
        help_text = """
🤖 **Demo Agent - Interactive Testing**

Available commands:
• `/help` - Show this help message
• `/echo <text>` - Echo back your text
• `/stream <text>` - Stream the response in chunks
• `/error` - Trigger an error to test error handling
• `/ask` - Request additional input (tests input_required state)
• `/background <task>` - Run a background task with progress updates
• `/calc <expression>` - Calculate a math expression (e.g., "2 + 3")
• `/memory` - Show conversation history
• `/context` - Show current context information
• `/slow <text>` - Respond slowly (tests timeout handling)
• `/multipart` - Return a multipart response with multiple artifacts

**Normal conversation:** Just chat normally to test multi-turn conversations!

Examples:
```
/echo Hello, world!
/stream Tell me a story
/calc 15 * 7
What is the capital of France?
Remember my name is Alice
What did I just tell you?
```

This agent demonstrates all MCP ↔ A2A bridge features for manual testing.
        """.strip()

        await event_queue.enqueue_event(new_agent_text_message(help_text))

    async def _handle_echo(self, text: str, event_queue: EventQueue) -> None:
        """Echo back the text."""
        # Remove the /echo command
        echo_text = re.sub(r"^/echo\s*", "", text, flags=re.IGNORECASE).strip()
        if not echo_text:
            echo_text = "(empty)"

        await event_queue.enqueue_event(new_agent_text_message(f"Echo: {echo_text}"))

    async def _handle_stream(self, text: str, event_queue: EventQueue) -> None:
        """Stream the response in chunks."""
        from a2a.types import TaskArtifactUpdateEvent

        # Remove the /stream command
        stream_text = re.sub(r"^/stream\s*", "", text, flags=re.IGNORECASE).strip()
        if not stream_text:
            stream_text = "This is a streaming response demonstration."

        # Split into chunks and stream
        words = stream_text.split()
        chunk_size = max(1, len(words) // 5)  # ~5 chunks

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)

            # Note: We need task_id and context_id from the context
            # For now, just send text messages
            await event_queue.enqueue_event(new_agent_text_message(chunk_text + " "))
            await asyncio.sleep(0.3)  # Simulate streaming delay

    async def _handle_error(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Trigger an error."""
        from a2a.types import TaskStatusUpdateEvent

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=context.task_id,
                contextId=context.context_id,
                status=TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message(
                        "Intentional error for testing error handling. "
                        "This demonstrates how A2A errors are mapped to MCP error codes."
                    ),
                ),
                final=True,
            )
        )

    async def _handle_ask(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Request additional input (input_required state)."""
        from a2a.types import TaskStatusUpdateEvent

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=context.task_id,
                contextId=context.context_id,
                status=TaskStatus(
                    state=TaskState.input_required,
                    message=new_agent_text_message(
                        "I need more information to proceed. Please provide:\n"
                        "1. Your preferred color\n"
                        "2. A number between 1-10\n\n"
                        "Reply with: CONFIRM: <color> <number>"
                    ),
                ),
                final=False,
            )
        )

    async def _handle_background(
        self, text: str, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Run a background task with progress updates."""
        from a2a.types import TaskStatusUpdateEvent

        task_name = re.sub(r"^/background\s*", "", text, flags=re.IGNORECASE).strip()
        if not task_name:
            task_name = "data processing"

        # Send progress updates
        for i in range(5):
            progress = (i + 1) * 20
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    taskId=context.task_id,
                    contextId=context.context_id,
                    status=TaskStatus(
                        state=TaskState.working,
                        message=new_agent_text_message(
                            f"Background task '{task_name}' progress: {progress}%"
                        ),
                    ),
                    final=False,
                )
            )
            await asyncio.sleep(0.5)

        # Final completion
        await event_queue.enqueue_event(
            new_agent_text_message(f"✓ Background task '{task_name}' completed successfully!")
        )

    async def _handle_calc(self, text: str, event_queue: EventQueue) -> None:
        """Calculate a math expression."""
        expr = re.sub(r"^/calc\s*", "", text, flags=re.IGNORECASE).strip()

        try:
            # Very simple calculator - just look for patterns
            match = re.search(r"(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)", expr)
            if match:
                a, op, b = match.groups()
                a, b = float(a), float(b)

                if op == "+":
                    result = a + b
                elif op == "-":
                    result = a - b
                elif op == "*":
                    result = a * b
                elif op == "/":
                    result = a / b if b != 0 else "Error: Division by zero"

                await event_queue.enqueue_event(
                    new_agent_text_message(f"Calculation: {a} {op} {b} = {result}")
                )
            else:
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        f"I can calculate simple expressions like '2 + 3' or '10 * 5'. "
                        f"I didn't understand: {expr}"
                    )
                )
        except Exception as e:
            await event_queue.enqueue_event(new_agent_text_message(f"Calculation error: {e}"))

    async def _handle_memory(self, context_id: str, event_queue: EventQueue) -> None:
        """Show conversation history."""
        history = self.memory.get_history(context_id)

        if not history:
            await event_queue.enqueue_event(
                new_agent_text_message("No conversation history for this context.")
            )
            return

        lines = ["**Conversation History:**\n"]
        for i, msg in enumerate(history[-10:], 1):  # Last 10 messages
            timestamp = msg["timestamp"]
            lines.append(f"{i}. [{timestamp}] {msg['role']}: {msg['text'][:100]}")

        await event_queue.enqueue_event(new_agent_text_message("\n".join(lines)))

    async def _handle_context(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Show current context information."""
        info = {
            "task_id": context.task_id,
            "context_id": context.context_id,
            "message_id": getattr(context.message, "messageId", None),
        }

        context_text = "**Current Context:**\n" + json.dumps(info, indent=2)
        await event_queue.enqueue_event(new_agent_text_message(context_text))

    async def _handle_slow(self, text: str, event_queue: EventQueue) -> None:
        """Respond slowly to test timeout handling."""
        slow_text = re.sub(r"^/slow\s*", "", text, flags=re.IGNORECASE).strip()

        await event_queue.enqueue_event(
            new_agent_text_message("Processing slowly... (testing timeout handling)")
        )
        await asyncio.sleep(2)
        await event_queue.enqueue_event(new_agent_text_message(f"Slow response: {slow_text}"))

    async def _handle_multipart(self, event_queue: EventQueue) -> None:
        """Return a multipart response."""
        parts = [
            "Part 1: This is a multipart response.",
            "Part 2: Each part is sent separately.",
            "Part 3: This tests artifact handling.",
        ]

        for part in parts:
            await event_queue.enqueue_event(new_agent_text_message(part))
            await asyncio.sleep(0.2)

    async def _handle_conversation(
        self, text: str, context_id: str, event_queue: EventQueue
    ) -> None:
        """Handle normal conversation with context awareness."""
        # Get conversation history
        history = self.memory.get_history(context_id)

        # Check for context-aware queries
        if any(word in text.lower() for word in ["remember", "told you", "said", "mentioned"]):
            if len(history) > 1:
                prev_messages = [msg["text"] for msg in history[-3:-1]]
                response = (
                    f"Looking at our recent conversation, you mentioned: "
                    f"{' | '.join(prev_messages[-2:])}. "
                    f"I'm keeping track of our conversation context!"
                )
            else:
                response = "This is the start of our conversation. I'll remember what we discuss!"
        elif "hello" in text.lower() or "hi" in text.lower():
            if len(history) > 1:
                response = "Hello again! We've been chatting for a bit now. How can I help you further?"
            else:
                response = "Hello! I'm the demo agent. I can help you test the MCP bridge. Try /help to see what I can do!"
        elif "?" in text:
            # Question - provide a helpful response
            response = (
                f"That's an interesting question! In a real agent, I would provide a detailed answer. "
                f"For testing purposes, I'm demonstrating multi-turn conversation capabilities. "
                f"Try asking me about something we discussed earlier, or use /memory to see our history."
            )
        else:
            # Generic response
            response = (
                f"I received your message: '{text[:50]}...'. "
                f"This is turn #{len(history)} in our conversation. "
                f"I'm maintaining context across multiple turns. Try /help for more capabilities!"
            )

        await event_queue.enqueue_event(new_agent_text_message(response))
