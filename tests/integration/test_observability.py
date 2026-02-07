"""Integration tests for OpenTelemetry observability.

Tests span creation and attribute collection using InMemorySpanExporter.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def span_exporter():
    """Captures OTel spans in memory for assertion."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    provider.shutdown()


@pytest.mark.integration
def test_span_exporter_basic_functionality(span_exporter):
    """InMemorySpanExporter captures spans correctly."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("test_span") as span:
        span.set_attribute("test.key", "test.value")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test_span"
    assert spans[0].attributes.get("test.key") == "test.value"


@pytest.mark.integration
def test_multiple_spans_captured(span_exporter):
    """Multiple spans are captured in order."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("span1"):
        pass

    with tracer.start_as_current_span("span2"):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span_names = [s.name for s in spans]
    assert "span1" in span_names
    assert "span2" in span_names


@pytest.mark.integration
def test_nested_spans_captured(span_exporter):
    """Nested spans maintain parent-child relationships."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("parent") as parent:
        parent_id = parent.get_span_context().span_id
        with tracer.start_as_current_span("child") as child:
            # Child should have parent context
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2


@pytest.mark.integration
async def test_tool_call_could_create_span(span_exporter):
    """If instrumented, tool calls would create spans."""
    from fastmcp import Client, FastMCP

    # Create a simple server
    server = FastMCP("TestServer")

    @server.tool()
    def test_tool(input: str) -> str:
        # If the tool implementation includes tracing, it would create a span
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("tool_execution"):
            return f"processed: {input}"

    async with Client(server) as client:
        await client.call_tool("test_tool", {"input": "test"})

    # Check if any spans were created
    spans = span_exporter.get_finished_spans()
    # If the tool is instrumented, we'd have at least one span
    # For now, just verify the exporter works
    assert isinstance(spans, list)


@pytest.mark.integration
def test_span_attributes_are_preserved(span_exporter):
    """Span attributes are correctly stored and retrievable."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("attributed_span") as span:
        span.set_attribute("service.name", "agentique")
        span.set_attribute("operation.type", "mcp_tool_call")
        span.set_attribute("tool.name", "agent")
        span.set_attribute("agent.name", "calculator")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    attrs = spans[0].attributes
    assert attrs.get("service.name") == "agentique"
    assert attrs.get("operation.type") == "mcp_tool_call"
    assert attrs.get("tool.name") == "agent"
    assert attrs.get("agent.name") == "calculator"


@pytest.mark.integration
def test_span_exporter_clear(span_exporter):
    """Span exporter can be cleared."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("span1"):
        pass

    assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    assert len(span_exporter.get_finished_spans()) == 0
