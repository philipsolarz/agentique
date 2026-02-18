"""LLM-powered human emulator for simulated conversations.

The SimulationAgent acts as a simulated human user, having natural conversations
with AI agents through the full MCP pipeline. It generates realistic messages,
reads responses, and analyzes the conversation for insights.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from .models import (
    ConversationTurn,
    EventType,
    SessionEvent,
    SimulationConfig,
    SimulationInsight,
    SimulationResult,
    SimulationState,
)

logger = logging.getLogger(__name__)

OnEventCallback = Callable[[SessionEvent], Awaitable[None]] | None


class SimulationAgent:
    """Emulates a human user having a natural conversation with AI agents.

    Uses an LLM to generate realistic human messages, decide when to continue
    or stop the conversation, and analyze the interaction for insights.
    """

    def __init__(
        self,
        client: Any,  # MCPTestClient
        config: SimulationConfig,
        on_event: OnEventCallback = None,
        reasoning_client: Any = None,  # MCPTestClient for Cogito reasoning
    ) -> None:
        self.client = client
        self.config = config
        self.on_event = on_event
        self._reasoning_client: Any = reasoning_client

        self.simulation_id = str(uuid.uuid4())[:8]
        self.state = SimulationState.IDLE
        self.conversation: list[ConversationTurn] = []
        self.insights: list[SimulationInsight] = []
        self.events: list[SessionEvent] = []

        # Control signals
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._stop_requested = False

        # LLM client
        self._llm_client: Any = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize the LLM client."""
        if self.config.llm_provider == "gemini":
            try:
                import google.generativeai as genai

                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not set")

                genai.configure(api_key=api_key)
                self._llm_client = genai.GenerativeModel(self.config.llm_model)
                logger.info("Initialized Gemini model: %s", self.config.llm_model)
            except ImportError:
                logger.error("google-generativeai not installed")
                raise
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")

    async def _call_cogito(self, prompt: str) -> str:
        """Route reasoning through the MCP/A2A pipeline to Cogito.

        Sends the prompt as a message through the reasoning client,
        which goes through MCP → A2A → Cogito agent, creating a true
        mirror where Cogito reasons with itself.
        """
        if not self._reasoning_client:
            raise RuntimeError("Reasoning client not available for Cogito mode")

        try:
            response = await self._reasoning_client.send_message(
                prompt, timeout=self.config.mcp_timeout,
            )
            return response.get("text", "")
        except Exception as exc:
            logger.warning("Cogito reasoning call failed, falling back to LLM: %s", exc)
            return await self._call_llm(prompt)

    def _get_reasoning_fn(self):
        """Return the appropriate reasoning function based on config."""
        if self.config.use_cogito_reasoning and self._reasoning_client:
            return self._call_cogito
        return self._call_llm

    async def run(self) -> SimulationResult:
        """Run the full simulation conversation loop.

        Returns:
            SimulationResult with conversation, insights, and summary
        """
        self.state = SimulationState.RUNNING
        self._stop_requested = False
        start_time = time.monotonic()

        await self._emit(EventType.SIMULATION_STARTED, {
            "persona": self.config.persona.model_dump(),
            "objective": self.config.objective.model_dump(),
        })

        try:
            # Generate opening message
            opening = await self._generate_opening_message()
            turn_number = 0

            for turn_idx in range(self.config.objective.max_turns):
                if self._stop_requested:
                    self.state = SimulationState.STOPPED
                    await self._emit(EventType.SIMULATION_STOPPED, {
                        "reason": "User requested stop",
                        "turns_completed": turn_number,
                    })
                    break

                # Check pause
                await self._pause_event.wait()

                # Check duration limit
                elapsed = time.monotonic() - start_time
                if elapsed > self.config.objective.max_duration_seconds:
                    await self._emit(EventType.SIMULATION_STOPPED, {
                        "reason": "Duration limit reached",
                        "turns_completed": turn_number,
                    })
                    break

                turn_number = turn_idx + 1
                message = opening if turn_idx == 0 else await self._generate_next_message()

                if not message:
                    break

                # Simulate thinking
                await self._emit(EventType.THINKING, {
                    "persona": self.config.persona.name,
                    "turn": turn_number,
                })

                # Simulate typing
                typing_time_ms = len(message) * self.config.persona.typing_speed_ms
                await self._emit(EventType.TYPING, {
                    "message": message,
                    "typing_speed_ms": self.config.persona.typing_speed_ms,
                    "duration_ms": typing_time_ms,
                })
                await asyncio.sleep(min(typing_time_ms / 1000, 3.0))  # Cap at 3s

                # Send message through MCP
                await self._emit(EventType.MESSAGE_SENT, {
                    "role": "human",
                    "message": message,
                    "turn": turn_number,
                })

                human_turn = ConversationTurn(
                    turn_number=turn_number,
                    role="human",
                    message=message,
                    thinking_time_ms=typing_time_ms,
                )
                self.conversation.append(human_turn)

                # Wait for AI response
                await self._emit(EventType.WAITING, {"turn": turn_number})

                response_start = time.monotonic()
                try:
                    response = await self.client.send_message(
                        message, timeout=self.config.mcp_timeout,
                    )
                    response_text = response.get("text", "")
                    is_error = response.get("is_error", False)
                except Exception as exc:
                    response_text = f"Error: {exc}"
                    is_error = True

                response_time_ms = (time.monotonic() - response_start) * 1000

                await self._emit(EventType.RESPONSE_RECEIVED, {
                    "message": response_text,
                    "is_error": is_error,
                    "response_time_ms": response_time_ms,
                    "turn": turn_number,
                })

                agent_turn = ConversationTurn(
                    turn_number=turn_number,
                    role="agent",
                    message=response_text,
                    thinking_time_ms=response_time_ms,
                )
                self.conversation.append(agent_turn)

                # Simulate reading response
                reading_time_ms = len(response_text) * self.config.persona.reading_speed_ms
                await self._emit(EventType.READING, {
                    "duration_ms": reading_time_ms,
                    "turn": turn_number,
                })
                await asyncio.sleep(min(reading_time_ms / 1000, 2.0))  # Cap at 2s

                await self._emit(EventType.TURN_COMPLETE, {
                    "turn": turn_number,
                    "human_message": message,
                    "agent_response": response_text[:200],
                })

                # Decide whether to continue
                if turn_number < self.config.objective.max_turns:
                    should_continue = await self._should_continue()
                    if not should_continue:
                        break

            # Analyze conversation and generate insights
            if self.state != SimulationState.STOPPED:
                self.state = SimulationState.COMPLETED

            summary = await self._analyze_conversation()
            total_time_ms = (time.monotonic() - start_time) * 1000

            result = SimulationResult(
                simulation_id=self.simulation_id,
                config=self.config,
                state=self.state,
                conversation=self.conversation,
                insights=self.insights,
                total_time_ms=total_time_ms,
                summary=summary,
            )

            await self._emit(
                EventType.SIMULATION_COMPLETED if self.state == SimulationState.COMPLETED
                else EventType.SIMULATION_STOPPED,
                {
                    "summary": summary,
                    "turns": len(self.conversation) // 2,
                    "insights_count": len(self.insights),
                    "total_time_ms": total_time_ms,
                },
            )

            return result

        except Exception as exc:
            self.state = SimulationState.FAILED
            total_time_ms = (time.monotonic() - start_time) * 1000
            await self._emit(EventType.SIMULATION_FAILED, {
                "error": str(exc),
                "total_time_ms": total_time_ms,
            })

            # Best-effort: try to generate insights from partial conversation
            summary = f"Simulation failed: {exc}"
            if self.conversation:
                try:
                    summary = await self._analyze_conversation()
                except Exception as analysis_exc:
                    logger.warning("Best-effort analysis also failed: %s", analysis_exc)

            return SimulationResult(
                simulation_id=self.simulation_id,
                config=self.config,
                state=SimulationState.FAILED,
                conversation=self.conversation,
                insights=self.insights,
                total_time_ms=total_time_ms,
                summary=summary,
            )

    async def pause(self) -> None:
        """Pause the simulation."""
        self._pause_event.clear()
        self.state = SimulationState.PAUSED
        await self._emit(EventType.SIMULATION_PAUSED, {
            "turns_completed": len(self.conversation) // 2,
        })

    async def resume(self) -> None:
        """Resume a paused simulation."""
        self._pause_event.set()
        self.state = SimulationState.RUNNING
        await self._emit(EventType.SIMULATION_RESUMED, {
            "turns_completed": len(self.conversation) // 2,
        })

    def stop(self) -> None:
        """Request the simulation to stop."""
        self._stop_requested = True
        # Also resume if paused so the loop can exit
        self._pause_event.set()

    async def _generate_opening_message(self) -> str:
        """Generate the first message based on persona and objective."""
        persona = self.config.persona
        objective = self.config.objective
        reason = self._get_reasoning_fn()

        topics_hint = ""
        if objective.topics:
            topics_hint = f"\nTopics to explore: {', '.join(objective.topics)}"

        if self.config.use_cogito_reasoning:
            prompt = f"""You are an autonomous improvement agent inside the Agentique self-improvement loop.

YOUR MISSION: Find a concrete bug, gap, or improvement in the Agentique codebase and produce an \
actionable fix. You are talking to CogitoPrime, which has 6 sub-agents including CodebaseExplorer \
that can read/search the actual source code.

Agentique is a protocol-agnostic bridge: MCP clients → Agentique → A2A/other protocols → agents.
The codebase is at src/agentique/ with: core/, bridge/, adapters/a2a/, simulation/, server.py

KNOWN IMPROVEMENT TARGETS (from the project roadmap):
- Phase 4 NOT STARTED: entry-point adapter discovery, OpenAI Agents API adapter, agentique-core package
- Error handling: no typed exception hierarchy (AgentiqueError, AgentNotFoundError, etc.)
- ToolMapper: no pluggable protocol for how agent cards become MCP tools
- A2A error codes (-32001 TaskNotFound, -32002, -32003) not mapped to MCP errors
- Missing tests for edge cases in streaming, task cancellation, error propagation
- Context ID management: MCP session IDs not mapped to A2A context IDs

Goal: {objective.goal}
{topics_hint}

Generate an opening message that gives CogitoPrime a SPECIFIC TASK. Examples:
- "Read adapters/a2a/adapter.py and find where A2A error codes are handled. I suspect -32001/-32002/-32003 aren't mapped to proper MCP errors. Show me what's there and propose a fix."
- "Search for 'class.*Error' across the codebase. We need a typed exception hierarchy. Show me what exists and draft what's missing."
- "Read bridge/router.py. The ToolMapper should be a pluggable Protocol, not hardcoded. Find the tool creation logic and propose extracting it."

Be DIRECTIVE. Give a task, not a question. Push for a diff, not a discussion.

Return ONLY the message text, nothing else."""
        else:
            prompt = f"""You are simulating a human user named {persona.name} who is a {persona.role}.
Personality: {', '.join(persona.personality_traits)}
Conversation style: {persona.conversation_style}

Your goal: {objective.goal}
{topics_hint}

Generate a natural opening message that this person would type to start a conversation
with an AI assistant. Keep it short (1-2 sentences), casual, and natural.

Return ONLY the message text, nothing else."""

        return await reason(prompt)

    async def _generate_next_message(self) -> str:
        """Generate the next human message based on conversation history."""
        persona = self.config.persona
        objective = self.config.objective
        reason = self._get_reasoning_fn()

        conversation_text = "\n".join(
            f"{'Human' if t.role == 'human' else 'AI'}: {t.message}"
            for t in self.conversation
        )

        topics_hint = ""
        if objective.topics:
            remaining = [
                t for t in objective.topics
                if not any(t.lower() in turn.message.lower() for turn in self.conversation)
            ]
            if remaining:
                topics_hint = f"\nTopics still to explore: {', '.join(remaining)}"

        if self.config.use_cogito_reasoning:
            prompt = f"""You are an autonomous improvement agent in the Agentique self-improvement loop.
You're talking to CogitoPrime (has CodebaseExplorer that reads actual source files).

Conversation so far:
{conversation_text}

Goal: {objective.goal}
{topics_hint}

RULES — be aggressive about driving toward a concrete improvement:
1. If the agent gave a vague analysis → REJECT it. Say "Don't describe the architecture. Read [specific file] and show me the exact code that needs to change."
2. If the agent showed code → PUSH for the diff. Say "Good. Now write the concrete fix: show old code vs new code."
3. If the agent proposed a fix → CHALLENGE it. Ask "What breaks? What edge cases? Read the test file and check coverage."
4. If the agent found a real bug → ESCALATE. Say "This is a real issue. Write the exact code change as a before/after diff."
5. If the agent is stalling or repeating → PIVOT to a new target from the roadmap: error hierarchy, ToolMapper protocol, A2A error code mapping, missing tests, context ID management.
6. If the agent timed out → move to a different improvement target immediately.

Each turn should drive closer to a CONCRETE, IMPLEMENTABLE code change.
Never be satisfied with descriptions or architecture diagrams. Demand code.

Return ONLY the message text, nothing else."""
        else:
            prompt = f"""You are {persona.name}, a {persona.role}.
Personality: {', '.join(persona.personality_traits)}
Style: {persona.conversation_style}

Conversation so far:
{conversation_text}

Goal: {objective.goal}
{topics_hint}

Generate {persona.name}'s next message. Be natural, concise (1-3 sentences), and human-like.
React to what the AI said, ask follow-up questions, or explore new topics.

Return ONLY the message text, nothing else."""

        return await reason(prompt)

    async def _should_continue(self) -> bool:
        """Ask the LLM whether the conversation should continue."""
        reason = self._get_reasoning_fn()

        conversation_text = "\n".join(
            f"{'Human' if t.role == 'human' else 'AI'}: {t.message}"
            for t in self.conversation[-6:]  # Last 3 exchanges
        )

        if self.config.use_cogito_reasoning:
            prompt = f"""Given this conversation in the Agentique self-improvement loop:
{conversation_text}

Turns so far: {len(self.conversation) // 2}
Max turns: {self.config.objective.max_turns}

Has a CONCRETE, IMPLEMENTABLE code change been produced? (actual before/after diff, not just description)
- If NO concrete diff has been produced yet → return "yes" (keep pushing)
- If a real code fix was proposed with specific changes → return "no" (objective met)
- If the conversation is stuck in loops or vague discussion → return "yes" (pivot to new target)

Return ONLY "yes" or "no"."""
        else:
            prompt = f"""Given this conversation:
{conversation_text}

Objective: {self.config.objective.goal}
Turns so far: {len(self.conversation) // 2}
Max turns: {self.config.objective.max_turns}

Should the human continue the conversation? Consider:
- Has the objective been met?
- Is there more to explore?
- Would a real human keep going?

Return ONLY "yes" or "no"."""

        response = await reason(prompt)
        return response.strip().lower().startswith("yes")

    async def _analyze_conversation(self) -> str:
        """Analyze the full conversation for insights and generate a summary."""
        if not self.conversation:
            return "No conversation to analyze."

        reason = self._get_reasoning_fn()

        conversation_text = "\n".join(
            f"{'Human' if t.role == 'human' else 'AI'}: {t.message}"
            for t in self.conversation
        )

        cogito_criteria = ""
        if self.config.use_cogito_reasoning:
            cogito_criteria = """
6. CONCRETE CODE CHANGES proposed — with file paths, function names, and before/after diffs
7. Roadmap gaps identified — Phase 4 items (entry-point discovery, OpenAI adapter, typed exceptions, ToolMapper protocol)
8. Was a real, implementable improvement produced? If yes, mark as category "improvement_proposal" with severity "high"
9. Did the conversation get stuck in abstract discussion without producing code? If yes, mark as category "process_issue"
"""

        prompt = f"""Analyze this conversation between a simulated human and an AI agent:

{conversation_text}

Identify:
1. Any bugs or errors in AI responses
2. UX issues (confusing responses, slow responses, wrong format)
3. Capability gaps (things the AI couldn't do)
4. Edge cases discovered
5. Positive observations
{cogito_criteria}
Return JSON:
{{
  "summary": "Brief 2-3 sentence summary of the conversation",
  "insights": [
    {{
      "category": "bug|ux_issue|capability_gap|edge_case|observation",
      "description": "What was found",
      "severity": "critical|high|medium|low|info",
      "turn_number": 1,
      "evidence": ["specific observation"]
    }}
  ]
}}"""

        try:
            response = await reason(prompt)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                summary = data.get("summary", "")

                for insight_data in data.get("insights", []):
                    insight = SimulationInsight(**insight_data)
                    self.insights.append(insight)
                    await self._emit(EventType.INSIGHT, insight.model_dump())

                await self._emit(EventType.SUMMARY, {"summary": summary})
                return summary
        except Exception as e:
            logger.error("Failed to analyze conversation: %s", e)

        return "Conversation completed but analysis failed."

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt, using retry with exponential backoff."""
        if self.config.llm_provider != "gemini":
            raise NotImplementedError(f"Provider {self.config.llm_provider} not supported")

        last_exc: Exception | None = None
        for attempt in range(self.config.llm_max_retries + 1):
            try:
                response = await asyncio.to_thread(
                    self._llm_client.generate_content, prompt
                )
                return response.text
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable_error(exc) or attempt >= self.config.llm_max_retries:
                    raise
                delay = self.config.llm_retry_base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.config.llm_max_retries + 1, exc, delay,
                )
                await asyncio.sleep(delay)

        # Should not reach here, but satisfy type checker
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """Check if an exception is retryable (rate limit, server error, network)."""
        type_name = type(exc).__name__
        retryable_types = {
            "ResourceExhausted", "InternalServerError", "ServiceUnavailable",
            "TooManyRequests", "ConnectionError", "TimeoutError",
        }
        if type_name in retryable_types or isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        err_str = str(exc).lower()
        return "429" in err_str or "rate limit" in err_str or "503" in err_str or "500" in err_str

    async def _emit(self, event_type: EventType, data: dict | None = None) -> None:
        """Emit a simulation event."""
        event = SessionEvent(
            type=event_type,
            data=data or {},
            simulation_id=self.simulation_id,
        )
        self.events.append(event)
        if self.on_event:
            try:
                await self.on_event(event)
            except Exception:
                logger.exception("Error in on_event callback")
