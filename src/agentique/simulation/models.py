"""Pydantic models for the Agentique Simulation system."""

from __future__ import annotations

import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field


class SimulationPersona(BaseModel):
    """Defines a simulated human persona for conversations."""

    name: str = "Alex"
    role: str = "curious user"
    personality_traits: list[str] = Field(
        default_factory=lambda: ["friendly", "inquisitive", "concise"]
    )
    conversation_style: str = "casual and direct"
    typing_speed_ms: float = 80.0  # ms per character
    reading_speed_ms: float = 20.0  # ms per character

    @classmethod
    def cogito(cls, name: str = "Cogito") -> SimulationPersona:
        """Create a Cogito persona — autonomous improvement agent that drives code changes."""
        return cls(
            name=name,
            role="autonomous improvement agent in a self-improving development loop",
            personality_traits=[
                "relentlessly improvement-driven",
                "demands concrete code over discussion",
                "rejects vague architecture talk",
                "identifies bugs and gaps in real code",
                "pushes for implementable diffs",
                "knows the Agentique roadmap and its gaps",
            ],
            conversation_style="directive, demanding, and diff-oriented",
            typing_speed_ms=120.0,
            reading_speed_ms=30.0,
        )


class SimulationObjective(BaseModel):
    """What the simulation should accomplish."""

    goal: str = "Have a natural conversation exploring the AI agent's capabilities"
    max_turns: int = 10
    max_duration_seconds: float = 600.0
    topics: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)


class SimulationConfig(BaseModel):
    """Full configuration for a simulation run."""

    persona: SimulationPersona = Field(default_factory=SimulationPersona)
    objective: SimulationObjective = Field(default_factory=SimulationObjective)
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.0-flash"
    mcp_url: str = "http://localhost:8000/mcp"
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 2.0
    mcp_timeout: float = 300.0  # seconds; deep reasoning agents need >30s
    use_cogito_reasoning: bool = False
    reasoning_mcp_url: str | None = None


class ConversationTurn(BaseModel):
    """A single turn in the simulated conversation."""

    turn_number: int
    role: str  # "human" or "agent"
    message: str
    timestamp: float = Field(default_factory=time.time)
    thinking_time_ms: float = 0.0


class SimulationState(str, Enum):
    """State of a simulation run."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class SimulationInsight(BaseModel):
    """An insight discovered during simulation."""

    category: str  # "bug", "ux_issue", "capability_gap", "edge_case", "observation"
    description: str
    severity: str = "info"  # "critical", "high", "medium", "low", "info"
    turn_number: int | None = None
    evidence: list[str] = Field(default_factory=list)


class SimulationResult(BaseModel):
    """Result of a completed simulation."""

    simulation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: SimulationConfig = Field(default_factory=SimulationConfig)
    state: SimulationState = SimulationState.COMPLETED
    conversation: list[ConversationTurn] = Field(default_factory=list)
    insights: list[SimulationInsight] = Field(default_factory=list)
    total_time_ms: float = 0.0
    summary: str = ""


class EventType(str, Enum):
    """Event types for the simulation system."""

    # MCP protocol events (kept from testing)
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LOG_MESSAGE = "log_message"
    PROGRESS = "progress"
    ERROR = "error"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"

    # Simulation lifecycle
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_STOPPED = "simulation_stopped"
    SIMULATION_COMPLETED = "simulation_completed"
    SIMULATION_FAILED = "simulation_failed"
    SIMULATION_PAUSED = "simulation_paused"
    SIMULATION_RESUMED = "simulation_resumed"

    # Conversation flow
    THINKING = "thinking"
    TYPING = "typing"
    MESSAGE_SENT = "message_sent"
    WAITING = "waiting"
    RESPONSE_RECEIVED = "response_received"
    READING = "reading"
    TURN_COMPLETE = "turn_complete"

    # Analysis
    INSIGHT = "insight"
    SUMMARY = "summary"


class SessionEvent(BaseModel):
    """An event emitted during simulation."""

    type: EventType
    timestamp: float = Field(default_factory=time.time)
    data: dict = Field(default_factory=dict)
    simulation_id: str | None = None
