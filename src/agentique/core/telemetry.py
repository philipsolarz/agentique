"""OpenTelemetry integration for agentique.

Adds ``agentique.*`` span attributes to FastMCP's built-in tracing.
Uses only ``opentelemetry-api`` (no-op if the SDK is not installed).

Attributes emitted:

    - ``agentique.agent_name``  — target agent name
    - ``agentique.protocol``    — adapter protocol (a2a, http, …)
    - ``agentique.task_state``  — last known task state
    - ``agentique.task_id``     — bridge task ID

Usage::

    from agentique.core.telemetry import trace_agent_call

    with trace_agent_call("my-agent", protocol="a2a") as span:
        # ... do work ...
        span.set_attribute("agentique.task_state", "completed")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

INSTRUMENTATION_NAME = "agentique"

_tracer: Any | None = None


def get_tracer() -> Any:
    """Return an agentique OTel tracer (no-op if SDK not installed)."""
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace

        _tracer = trace.get_tracer(INSTRUMENTATION_NAME)
    except ImportError:
        _tracer = _NoOpTracer()
    return _tracer


@contextmanager
def trace_agent_call(
    agent_name: str,
    *,
    protocol: str = "unknown",
    task_id: str | None = None,
) -> Iterator[Any]:
    """Context manager that creates an OTel span with agentique attributes.

    If the OpenTelemetry SDK is not installed this is a no-op.
    """
    tracer = get_tracer()
    try:
        span = tracer.start_span(f"agentique.call.{agent_name}")
    except Exception:
        yield _NoOpSpan()
        return

    try:
        span.set_attribute("agentique.agent_name", agent_name)
        span.set_attribute("agentique.protocol", protocol)
        if task_id:
            span.set_attribute("agentique.task_id", task_id)
        yield span
    except Exception as exc:
        try:
            span.set_attribute("agentique.error", str(exc))
            span.set_status(2, str(exc))  # StatusCode.ERROR = 2
        except Exception:
            pass
        raise
    finally:
        try:
            span.end()
        except Exception:
            pass


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current active span (if any)."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute(key, value)
    except ImportError:
        pass


class _NoOpTracer:
    """Fallback tracer when OpenTelemetry is not installed."""

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpSpan:
    """Fallback span that silently discards all operations."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, code: int, description: str = "") -> None:
        pass

    def end(self) -> None:
        pass

    def is_recording(self) -> bool:
        return False
