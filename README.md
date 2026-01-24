# Agentique

Bridge the Model Context Protocol (MCP) with the Agent2Agent (A2A) protocol.

## Overview

Agentique runs a FastMCP 3.0 server that routes MCP tool calls to A2A agents. The server is intentionally narrow in scope:

- MCP handles protocol surface (tools, resources, prompts, streaming).
- An `AgentRouter` selects the appropriate A2A agent.
- A lightweight A2A bridge sends messages and normalizes responses.

## Quick start

```bash
export AGENTIQUE_AGENTS="echo=http://localhost:9999|general"
python -m agentique
```

By default the server uses STDIO transport. Use HTTP transport by setting:

```bash
export AGENTIQUE_TRANSPORT=http
export AGENTIQUE_HOST=127.0.0.1
export AGENTIQUE_PORT=8000
```

## Programmatic usage

```python
from agentique import AgentDescriptor, create_server

agents = [
    AgentDescriptor(name="echo", base_url="http://localhost:9999", skills=("general",)),
]

mcp = create_server(agents=agents)

if __name__ == "__main__":
    mcp.run()
```

## MCP surface

Tools:

- `a2a_send`: Send a message to an A2A agent and return a structured response.
- `a2a_stream`: Stream agent responses as chunks.
- `a2a_list_agents`: List configured agents.

Resources:

- `a2a://agents`
- `a2a://agents/{agent}`
- `a2a://agents/{agent}/card`

Prompts:

- `a2a_routing_prompt`

## Local A2A test agent (Google ADK)

The enhanced multi-agent test server lives in `a2a_test_agent/` and demonstrates a sophisticated agent hierarchy with multiple specialized capabilities.

### Test Agent Architecture

The test agent includes:
- **TestAgentRoot**: Orchestrator that routes to specialized subagents
- **Calculator**: Arithmetic operations and statistical analysis
- **DataProcessor**: List filtering, sorting, and batch processing
- **TextProcessor**: Text transformations and keyword extraction
- **InfoRetriever**: Information lookup and simulated data fetching

See [a2a_test_agent/README.md](a2a_test_agent/README.md) for detailed documentation.

### Quick Setup

1. Copy `.env.example` to `.env` and add your Google API key:
```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_actual_key
```

2. Start both services with Docker Compose:
```bash
docker compose up --build
```

This starts:
- A2A server: http://localhost:9000 (agent card at `/.well-known/agent-card.json`)
- MCP server: http://localhost:8000

### Manual Setup (without Docker)

Start the A2A server:

```bash
cd a2a_test_agent
uv pip install -e .

export A2A_HOST=127.0.0.1
export A2A_PORT=9000
export A2A_BASE_URL=http://127.0.0.1:9000
export GOOGLE_API_KEY=your_api_key_here  # REQUIRED
export ADK_MODEL=gemini-2.0-flash

uv run adk-test-agent
```

Start the MCP server (in a separate terminal):

```bash
cd ..
uv pip install -e '.[a2a]'
export AGENTIQUE_AGENTS="root=http://127.0.0.1:9000|calculator,data_processing,text_manipulation,info_retrieval"
uv run python -m agentique
```

## Testing

Tests use FastMCP in-memory transport with an in-process A2A FastAPI app backed by a Google ADK agent. To run tests you will need:

- The A2A Python SDK (import path: `a2a`).
- Google ADK (import path: `google.adk`).
- `pytest`, `pytest-asyncio`, `httpx`, and `fastapi`.

Install dependencies with uv and run tests with:

```bash
uv pip install -e '.[a2a,adk,test]'
pytest
```

## Docker compose (fastest end-to-end setup)

```bash
docker compose up --build
```

- A2A server: http://localhost:9000 (agent card at `/.well-known/agent-card.json`)
- MCP server: http://localhost:8000
