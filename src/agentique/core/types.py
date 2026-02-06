"""Shared data structures used across all agentique layers.

These are plain dataclasses / enums with no heavy external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Task state machine
# ---------------------------------------------------------------------------


class TaskState(str, Enum):
    """A2A-aligned task lifecycle states."""

    submitted = "submitted"
    working = "working"
    input_required = "input-required"
    completed = "completed"
    canceled = "canceled"
    failed = "failed"
    rejected = "rejected"
    auth_required = "auth-required"
    unknown = "unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.rejected,
        }

    @property
    def is_interruptible(self) -> bool:
        return self in {TaskState.input_required, TaskState.auth_required}


# ---------------------------------------------------------------------------
# Agent information
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentInfo:
    """Canonical description of an agent, independent of protocol."""

    name: str
    base_url: str
    description: str | None = None
    skills: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "description": self.description,
            "skills": list(self.skills),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Bridge context (MCP session snapshot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BridgeContext:
    """Snapshot of MCP session state forwarded to adapters."""

    session_id: str | None = None
    request_id: str | None = None
    client_id: str | None = None
    meta: dict[str, Any] | None = None
    conversation_history: list[dict[str, str]] | None = None

    @classmethod
    def from_fastmcp_context(cls, ctx: Any) -> BridgeContext:
        """Build from a FastMCP ``Context`` object."""
        request_context = getattr(ctx, "request_context", None)
        meta = getattr(request_context, "meta", None) if request_context else None
        return cls(
            session_id=getattr(ctx, "session_id", None),
            request_id=getattr(ctx, "request_id", None),
            client_id=getattr(ctx, "client_id", None),
            meta=meta,
        )

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.session_id:
            payload["session_id"] = self.session_id
        if self.request_id:
            payload["request_id"] = self.request_id
        if self.client_id:
            payload["client_id"] = self.client_id
        if self.meta is not None:
            payload["meta"] = self.meta
        return payload


# ---------------------------------------------------------------------------
# Events — semantic events from agent execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentEvent:
    """A single event from an agent execution."""

    kind: str  # message, artifact, status, task, event
    text: str | None = None
    raw: Any | None = None
    # Task tracking
    task_id: str | None = None
    context_id: str | None = None
    progress: float | None = None
    # Artifact
    artifact_id: str | None = None
    artifact_name: str | None = None
    # Sub-agent visibility
    branch: str | None = None
    author: str | None = None
    # State
    state: str | None = None
    is_final: bool = False
    # Tool confirmation
    requires_confirmation: bool = False
    tool_call: dict[str, Any] | None = None
    # Extension metadata
    event_metadata: dict[str, Any] | None = None

    @property
    def is_content(self) -> bool:
        return self.kind in {"message", "artifact"}

    @property
    def is_status_update(self) -> bool:
        return self.kind in {"status", "task"}

    @property
    def agent_path(self) -> list[str]:
        return self.branch.split(".") if self.branch else []

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind, "text": self.text}
        for attr in (
            "task_id", "context_id", "progress", "artifact_id",
            "artifact_name", "branch", "author", "state",
        ):
            val = getattr(self, attr)
            if val is not None:
                result[attr] = val
        if self.is_final:
            result["is_final"] = True
        if self.requires_confirmation:
            result["requires_confirmation"] = True
        if self.tool_call is not None:
            result["tool_call"] = self.tool_call
        if self.event_metadata:
            result["metadata"] = self.event_metadata
        return result


# ---------------------------------------------------------------------------
# Responses & stream chunks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentResponse:
    """Complete response from an agent."""

    agent: str
    text: str
    events: tuple[AgentEvent, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "text": self.text,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class StreamChunk:
    """Streaming chunk from an agent."""

    agent: str
    index: int
    kind: str
    text: str | None = None
    task_id: str | None = None
    context_id: str | None = None
    progress: float | None = None
    branch: str | None = None
    author: str | None = None
    state: str | None = None
    is_final: bool = False
    requires_confirmation: bool = False
    tool_call: dict[str, Any] | None = None

    @classmethod
    def from_event(cls, agent: str, index: int, event: AgentEvent) -> StreamChunk:
        return cls(
            agent=agent, index=index, kind=event.kind, text=event.text,
            task_id=event.task_id, context_id=event.context_id,
            progress=event.progress, branch=event.branch, author=event.author,
            state=event.state, is_final=event.is_final,
            requires_confirmation=event.requires_confirmation,
            tool_call=event.tool_call,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "agent": self.agent, "index": self.index,
            "kind": self.kind, "text": self.text,
        }
        for attr in ("task_id", "context_id", "progress", "branch", "author", "state"):
            val = getattr(self, attr)
            if val is not None:
                result[attr] = val
        if self.is_final:
            result["is_final"] = True
        if self.requires_confirmation:
            result["requires_confirmation"] = True
        if self.tool_call is not None:
            result["tool_call"] = self.tool_call
        return result


# ---------------------------------------------------------------------------
# Task tracker
# ---------------------------------------------------------------------------


@dataclass
class TaskTracker:
    """Tracks an A2A task through its lifecycle."""

    task_id: str
    context_id: str | None = None
    state: TaskState = TaskState.submitted
    events: list[AgentEvent] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    progress: float = 0.0
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    hierarchy: AgentHierarchy | None = None

    def transition(self, new_state: TaskState, message: str | None = None) -> bool:
        if self.state.is_terminal:
            return False
        self.state = new_state
        if message:
            self.message = message
        return True

    def add_event(self, event: AgentEvent) -> None:
        self.events.append(event)
        if event.progress is not None:
            self.progress = event.progress
        if event.state:
            try:
                self.transition(TaskState(event.state))
            except ValueError:
                pass

    def to_dict(self) -> dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "context_id": self.context_id,
            "state": self.state.value,
            "progress": self.progress,
            "message": self.message,
            "event_count": len(self.events),
            "artifact_count": len(self.artifacts),
            "metadata": self.metadata,
        }
        if self.hierarchy:
            result["hierarchy"] = self.hierarchy.to_dict()
        return result


# ---------------------------------------------------------------------------
# Agent hierarchy (sub-agent visibility)
# ---------------------------------------------------------------------------


@dataclass
class SubAgentInfo:
    name: str
    description: str | None = None
    skills: list[str] = field(default_factory=list)
    parent: str | None = None
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "skills": self.skills,
            "parent": self.parent,
            "depth": self.depth,
        }


@dataclass
class AgentHierarchy:
    root: str
    agents: dict[str, SubAgentInfo] = field(default_factory=dict)

    def add_agent(
        self,
        name: str,
        description: str | None = None,
        skills: list[str] | None = None,
        parent: str | None = None,
    ) -> None:
        depth = 0
        if parent and parent in self.agents:
            depth = self.agents[parent].depth + 1
        self.agents[name] = SubAgentInfo(
            name=name, description=description,
            skills=skills or [], parent=parent, depth=depth,
        )

    def get_path(self, name: str) -> list[str]:
        path: list[str] = []
        current: str | None = name
        while current:
            path.insert(0, current)
            info = self.agents.get(current)
            current = info.parent if info else None
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "agents": {n: i.to_dict() for n, i in self.agents.items()},
        }
