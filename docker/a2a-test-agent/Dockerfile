# Enhanced A2A Test Agent Dockerfile
# Multi-agent system built with Google ADK exposed via A2A protocol
FROM python:3.12-slim

WORKDIR /app

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY a2a_test_agent/pyproject.toml a2a_test_agent/README.md /app/
COPY a2a_test_agent/src /app/src

# Install dependencies using uv
RUN uv pip install --system -e .

# Set default environment variables
ENV A2A_HOST=0.0.0.0
ENV A2A_PORT=9000
ENV A2A_PROTOCOL=http
ENV ADK_MODEL=gemini-2.0-flash

# Expose the A2A server port
EXPOSE 9000

# Health check to verify the server is running
# The agent card should be available at /.well-known/agent-card.json
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/.well-known/agent-card.json')" || exit 1

# Run the A2A server
CMD ["adk-test-agent"]
