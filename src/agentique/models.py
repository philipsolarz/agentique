from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    kind: str
    text: str | None
    raw: Any | None = None
    # Additional structured event data
    task_id: str | None = None
    progress: float | None = None
    artifact_id: str | None = None
    event_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "text": self.text,
        }
        # Include optional fields if present
        if self.task_id is not None:
            result["task_id"] = self.task_id
        if self.progress is not None:
            result["progress"] = self.progress
        if self.artifact_id is not None:
            result["artifact_id"] = self.artifact_id
        if self.event_metadata:
            result["metadata"] = self.event_metadata
        return result


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
    agent: str
    index: int
    kind: str
    text: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "index": self.index,
            "kind": self.kind,
            "text": self.text,
        }
