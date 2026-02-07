"""Tests for agentique.core.telemetry."""

from __future__ import annotations

import pytest

from agentique.core.telemetry import (
    _NoOpSpan,
    _NoOpTracer,
    get_tracer,
    set_span_attribute,
    trace_agent_call,
)


def test_no_op_span_is_silent():
    """NoOpSpan should accept all operations without error."""
    span = _NoOpSpan()
    span.set_attribute("key", "value")
    span.set_status(2, "error")
    span.end()
    assert span.is_recording() is False


def test_no_op_tracer_creates_no_op_span():
    """NoOpTracer should create NoOpSpan instances."""
    tracer = _NoOpTracer()
    span = tracer.start_span("test")
    assert isinstance(span, _NoOpSpan)


def test_get_tracer_returns_something():
    """get_tracer() should return a tracer (NoOp if OTel not installed)."""
    tracer = get_tracer()
    assert tracer is not None
    # Calling again should return the cached instance
    tracer2 = get_tracer()
    assert tracer2 is tracer


def test_trace_agent_call_context_manager():
    """trace_agent_call should work as a context manager."""
    with trace_agent_call("test-agent", protocol="a2a", task_id="task-1") as span:
        span.set_attribute("agentique.task_state", "working")
    # No exception → success


def test_trace_agent_call_propagates_exception():
    """trace_agent_call should re-raise exceptions from the body."""
    with pytest.raises(ValueError, match="test error"):
        with trace_agent_call("test-agent") as span:
            raise ValueError("test error")


def test_set_span_attribute_no_otel():
    """set_span_attribute should be a no-op when OTel is not available."""
    # Should not raise even when OTel is not installed
    set_span_attribute("agentique.test", "value")
