# Agentique

**Bridge any agent ecosystem to MCP.**

Agentique is a Python framework that exposes any agent ecosystem as a first-class MCP server. It translates MCP's tool/resource/prompt primitives into agent protocol operations, letting any MCP client — Claude, Cursor, ChatGPT, or custom hosts — seamlessly interact with remote agents regardless of their underlying protocol.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  MCP Clients (Claude, Cursor, custom)                   │
└────────────────────────┬────────────────────────────────┘
                         │  MCP Protocol
┌────────────────────────▼────────────────────────────────┐
│  Protocol Layer — FastMCP 3.0 Server                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │  agent   │ │  agents  │ │   task   │ │  inspect  │  │
│  └────┬─────┘ └──────────┘ └──────────┘ └───────────┘  │
│       │                                                 │
│  Bridge Layer — Routing, Translation, State             │
│  ┌──────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │  Router  │ │  Provider    │ │   Task Manager      │  │
│  └────┬─────┘ └──────────────┘ └─────────────────────┘  │
│       │                                                 │
│  Adapter Layer — Pluggable Backends                     │
│  ┌──────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │   A2A    │ │  (OpenAI)    │ │   (Custom HTTP)     │  │
│  └────┬─────┘ └──────────────┘ └─────────────────────┘  │
└───────┼─────────────────────────────────────────────────┘
        │  A2A Protocol
┌───────▼─────────────────────────────────────────────────┐
│  Agent Servers (ADK, LangChain, custom)                 │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install agentique
```

### As an MCP server (stdio)

```bash
export AGENTIQUE_AGENTS="myagent=http://localhost:9000|skill1,skill2"
agentique
```

### As an HTTP server

```bash
export AGENTIQUE_TRANSPORT=http
export AGENTIQUE_PORT=8000
export AGENTIQUE_AGENTS="myagent=http://localhost:9000|skill1,skill2"
agentique
```

### Programmatic usage

```python
from agentique import AgentInfo, create_server

server = create_server(agents=[
    AgentInfo(
        name="my-agent",
        base_url="http://localhost:9000",
        skills=("math", "text"),
    ),
])
server.run(transport="stdio")
```

## MCP Tools Exposed

| Tool | Description |
|------|-------------|
| `agent` | Send a message to an agent (streams response) |
| `agents` | List all available agents |
| `task` | Query task state and progress |
| `inspect` | View agent's sub-agent hierarchy |
| `agent_background` | Run agent task in background |

Each registered agent also gets its own direct tool, plus any tools declared in its A2A agent card.

## Key Concepts

### Protocol Classes (not ABCs)

All interfaces use `typing.Protocol` for structural subtyping:

```python
from agentique import AgentAdapter

# Any class with these methods is a valid adapter — no inheritance needed
class MyAdapter:
    async def discover_agents(self) -> list[AgentInfo]: ...
    async def send_message(self, agent_id, message, context) -> AgentResponse: ...
    async def stream_message(self, agent_id, message, context) -> AsyncIterator[AgentEvent]: ...
    async def close(self) -> None: ...

assert isinstance(MyAdapter(), AgentAdapter)  # True via structural subtyping
```

### Pluggable Routing

```python
from agentique.bridge.router import AgentRouter, KeywordRouter, DirectRouter

# Keyword-based (default) — matches message content to agent skills
router = AgentRouter(agents, strategy=KeywordRouter())

# Direct — always routes to a specific agent
router = AgentRouter(agents, strategy=DirectRouter("my-agent"))
```

### Typed Error Hierarchy

```python
from agentique import AgentNotFoundError, TaskNotFoundError

# Each error carries MCP and A2A error codes
try:
    router.describe("nonexistent")
except AgentNotFoundError as e:
    print(e.mcp_code)   # -32602
    print(e.a2a_code)   # -32001
```

### Lifecycle Events

```python
from agentique import AsyncEventEmitter

emitter = AsyncEventEmitter()

@emitter.on("task.created")
async def on_task(task_id: str):
    print(f"Task started: {task_id}")

server = create_server(agents=agents, events=emitter)
```

### Configuration via Environment

All settings sourced from `AGENTIQUE_*` env vars via Pydantic Settings:

```bash
AGENTIQUE_NAME=MyBridge
AGENTIQUE_TRANSPORT=http
AGENTIQUE_HOST=0.0.0.0
AGENTIQUE_PORT=8000
AGENTIQUE_CACHE_TTL=600
AGENTIQUE_AGENTS=agent1=http://host:9000|skill1,skill2
```

## Package Structure

```
src/agentique/
├── core/                  # Zero-dependency protocols, types, config
│   ├── protocols.py       # AgentAdapter, ToolMapper, BridgeMiddleware
│   ├── types.py           # AgentInfo, AgentEvent, TaskState, etc.
│   ├── config.py          # Pydantic Settings
│   ├── errors.py          # Typed exception hierarchy
│   └── events.py          # AsyncEventEmitter
├── bridge/                # Protocol-agnostic routing and state
│   ├── provider.py        # FastMCP 3.0 Provider
│   ├── router.py          # Pluggable routing strategies
│   └── task_manager.py    # Task lifecycle management
├── adapters/              # Protocol-specific backends
│   └── a2a/               # A2A adapter (first backend)
│       ├── adapter.py     # A2AAgentAdapter
│       ├── client.py      # A2A SDK client pool
│       └── card_parser.py # Agent card → MCP components
├── server.py              # FastMCP server factory
└── __main__.py            # CLI entry point
```

## Docker Compose

```bash
export GOOGLE_API_KEY=your_key
docker compose up --build
```

This starts the A2A test agent (port 9000) and MCP server (port 8000).

## Development

```bash
# Install
pip install -e ".[test]"

# Test
pytest tests/

# Run locally
export AGENTIQUE_AGENTS="root=http://localhost:9000|calculator,text"
python -m agentique
```

## License

MIT
