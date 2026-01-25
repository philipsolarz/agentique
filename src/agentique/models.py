from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskState(str, Enum):
    """A2A task lifecycle states aligned with the A2A protocol."""

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
        """Check if this is a terminal state."""
        return self in {
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.rejected,
        }

    @property
    def is_interruptible(self) -> bool:
        """Check if this state requires user intervention."""
        return self in {
            TaskState.input_required,
            TaskState.auth_required,
        }


@dataclass
class TaskTracker:
    """Track A2A task lifecycle and state transitions."""

    task_id: str
    context_id: str | None = None
    state: TaskState = TaskState.submitted
    events: list["AgentEvent"] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    progress: float = 0.0
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    hierarchy: "AgentHierarchy | None" = None  # Tracks sub-agent hierarchy observed during execution

    def transition(self, new_state: TaskState, message: str | None = None) -> bool:
        """Transition to a new state if valid."""
        if self.state.is_terminal:
            return False  # Cannot transition from terminal state
        self.state = new_state
        if message:
            self.message = message
        return True

    def add_event(self, event: "AgentEvent") -> None:
        """Add an event and update state accordingly."""
        self.events.append(event)

        # Update progress if available
        if event.progress is not None:
            self.progress = event.progress

        # Update state based on event kind
        if event.kind == "task":
            if event.text:
                self.message = event.text
            self.transition(TaskState.working)
        elif event.kind == "status":
            self._parse_status(event.text)
        elif event.kind in {"message", "artifact"}:
            if not self.state.is_terminal:
                self.transition(TaskState.completed)

    def _parse_status(self, status_text: str | None) -> None:
        """Parse status text to determine state."""
        if not status_text:
            return
        status_lower = status_text.lower()
        if "completed" in status_lower or "done" in status_lower:
            self.transition(TaskState.completed, status_text)
        elif "failed" in status_lower or "error" in status_lower:
            self.transition(TaskState.failed, status_text)
        elif "canceled" in status_lower or "cancelled" in status_lower:
            self.transition(TaskState.canceled, status_text)
        elif "input" in status_lower or "waiting" in status_lower:
            self.transition(TaskState.input_required, status_text)
        elif "auth" in status_lower:
            self.transition(TaskState.auth_required, status_text)
        else:
            self.transition(TaskState.working, status_text)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "context_id": self.context_id,
            "state": self.state.value,
            "progress": self.progress,
            "message": self.message,
            "event_count": len(self.events),
            "artifact_count": len(self.artifacts),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AgentDescriptor:
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


@dataclass(frozen=True)
class McpContextSnapshot:
    session_id: str | None
    request_id: str | None
    client_id: str | None
    meta: dict[str, Any] | None

    @classmethod
    def from_context(cls, ctx: Any) -> "McpContextSnapshot":
        request_context = getattr(ctx, "request_context", None)
        meta = None
        if request_context is not None:
            meta = getattr(request_context, "meta", None)
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


@dataclass(frozen=True)
class AgentEvent:
    """An event from an A2A agent with full semantic information."""

    kind: str
    text: str | None
    raw: Any | None = None
    # Task tracking
    task_id: str | None = None
    context_id: str | None = None
    progress: float | None = None
    # Artifact information
    artifact_id: str | None = None
    artifact_name: str | None = None
    # Sub-agent visibility (from ADK branch tracking)
    branch: str | None = None  # e.g., "root.calculator.add"
    author: str | None = None  # Which agent produced this event
    # State information
    state: str | None = None  # TaskState value
    is_final: bool = False
    # Tool confirmation support
    requires_confirmation: bool = False
    tool_call: dict[str, Any] | None = None
    # Extension metadata
    event_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "text": self.text,
        }
        # Include optional fields if present
        if self.task_id is not None:
            result["task_id"] = self.task_id
        if self.context_id is not None:
            result["context_id"] = self.context_id
        if self.progress is not None:
            result["progress"] = self.progress
        if self.artifact_id is not None:
            result["artifact_id"] = self.artifact_id
        if self.artifact_name is not None:
            result["artifact_name"] = self.artifact_name
        if self.branch is not None:
            result["branch"] = self.branch
        if self.author is not None:
            result["author"] = self.author
        if self.state is not None:
            result["state"] = self.state
        if self.is_final:
            result["is_final"] = self.is_final
        if self.requires_confirmation:
            result["requires_confirmation"] = self.requires_confirmation
        if self.tool_call is not None:
            result["tool_call"] = self.tool_call
        if self.event_metadata:
            result["metadata"] = self.event_metadata
        return result

    @property
    def is_content(self) -> bool:
        """Check if this event contains actual content."""
        return self.kind in {"message", "artifact"}

    @property
    def is_status_update(self) -> bool:
        """Check if this is a status update event."""
        return self.kind in {"status", "task"}

    @property
    def agent_path(self) -> list[str]:
        """Get the agent hierarchy path from branch."""
        if not self.branch:
            return []
        return self.branch.split(".")


@dataclass(frozen=True)
class AgentResponse:
    agent: str
    text: str
    events: tuple[AgentEvent, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "text": self.text,
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class StreamChunk:
    """A streaming chunk from an A2A agent."""

    agent: str
    index: int
    kind: str
    text: str | None
    # Task tracking
    task_id: str | None = None
    context_id: str | None = None
    progress: float | None = None
    # Sub-agent visibility
    branch: str | None = None
    author: str | None = None
    # State information
    state: str | None = None
    is_final: bool = False
    # Tool confirmation
    requires_confirmation: bool = False
    tool_call: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "agent": self.agent,
            "index": self.index,
            "kind": self.kind,
            "text": self.text,
        }
        if self.task_id is not None:
            result["task_id"] = self.task_id
        if self.context_id is not None:
            result["context_id"] = self.context_id
        if self.progress is not None:
            result["progress"] = self.progress
        if self.branch is not None:
            result["branch"] = self.branch
        if self.author is not None:
            result["author"] = self.author
        if self.state is not None:
            result["state"] = self.state
        if self.is_final:
            result["is_final"] = self.is_final
        if self.requires_confirmation:
            result["requires_confirmation"] = self.requires_confirmation
        if self.tool_call is not None:
            result["tool_call"] = self.tool_call
        return result

    @classmethod
    def from_event(cls, agent: str, index: int, event: "AgentEvent") -> "StreamChunk":
        """Create a StreamChunk from an AgentEvent."""
        return cls(
            agent=agent,
            index=index,
            kind=event.kind,
            text=event.text,
            task_id=event.task_id,
            context_id=event.context_id,
            progress=event.progress,
            branch=event.branch,
            author=event.author,
            state=event.state,
            is_final=event.is_final,
            requires_confirmation=event.requires_confirmation,
            tool_call=event.tool_call,
        )


@dataclass
class SubAgentInfo:
    """Information about a sub-agent in a multi-agent system."""

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
    """Represents the hierarchy of agents including sub-agents."""

    root: str
    agents: dict[str, SubAgentInfo] = field(default_factory=dict)

    def add_agent(
        self,
        name: str,
        description: str | None = None,
        skills: list[str] | None = None,
        parent: str | None = None,
    ) -> None:
        """Add an agent to the hierarchy."""
        depth = 0
        if parent and parent in self.agents:
            depth = self.agents[parent].depth + 1
        self.agents[name] = SubAgentInfo(
            name=name,
            description=description,
            skills=skills or [],
            parent=parent,
            depth=depth,
        )

    def get_path(self, name: str) -> list[str]:
        """Get the path from root to the named agent."""
        path = []
        current = name
        while current:
            path.insert(0, current)
            info = self.agents.get(current)
            if info:
                current = info.parent
            else:
                break
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "agents": {name: info.to_dict() for name, info in self.agents.items()},
        }


@dataclass
class ElicitationRequest:
    """Request for user input during agent execution."""

    message: str
    response_type: str = "text"  # text, boolean, choice
    choices: list[str] | None = None
    default: str | None = None
    task_id: str | None = None
    context_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "message": self.message,
            "response_type": self.response_type,
        }
        if self.choices:
            result["choices"] = self.choices
        if self.default:
            result["default"] = self.default
        if self.task_id:
            result["task_id"] = self.task_id
        if self.context_id:
            result["context_id"] = self.context_id
        return result


@dataclass
class ToolConfirmationRequest:
    """Request for tool execution confirmation."""

    tool_name: str
    arguments: dict[str, Any]
    description: str | None = None
    agent: str | None = None
    task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "description": self.description,
            "agent": self.agent,
            "task_id": self.task_id,
        }
