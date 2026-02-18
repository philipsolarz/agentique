# Agentique: AI Simulation & Development Loop

## Mission

**Agentique** is a protocol-agnostic bridge that exposes any agent ecosystem as a first-class MCP server. It enables seamless integration between MCP clients (Claude, Cursor, ChatGPT) and remote agents regardless of their underlying protocol (A2A, custom, etc).

**The simulation system** creates AI-powered human personas that have natural conversations with agents through the full MCP pipeline, discovering bugs, UX issues, and capability gaps — enabling a self-improving development loop.

## Architecture

```
Claude Code ──MCP──> Simulator MCP Server (stdio, local)
                            │
                            │ REST API
                            v
                     Simulation UI + Engine (port 8080)
                      ├─ Browser UI (WebSocket observer)
                      ├─ SimulationHarness (engine)
                      └─ SimulationAgent (Gemini-powered human emulator)
                            │
                            │ MCPTestClient
                            v
                     Agentique MCP Server (port 8000)
                            │
                            │ A2A Protocol
                            v
                     A2A Agent Server (port 9000)
```

## Quick Start

```bash
# 1. Set API key
echo "GOOGLE_API_KEY=your-key-here" > .env

# 2. Start simulation stack
docker compose -f docker-compose.simulation.yml up -d --wait

# 3. Open observer UI
xdg-open http://localhost:8080

# 4. Restart Claude Code to pick up MCP config, then:
#    "Start a simulation with a curious developer exploring math capabilities"
```

## Simulator MCP Tools

The Simulator MCP Server (`agentique-simulator` in `.claude/mcp.json`) exposes 6 tools:

| Tool | Description |
|------|-------------|
| `start_simulation` | Start a simulated conversation (objective, persona, style, topics, max_turns) |
| `stop_simulation` | Stop a running simulation |
| `get_simulation_status` | Get state, turns, insights count |
| `get_simulation_events` | Get event log with pagination |
| `get_simulation_insights` | Get discovered bugs/issues |
| `list_simulations` | List all active and completed simulations |

## Development

### Running Tests

```bash
uv run pytest tests/                              # All tests (333 passing)
uv run pytest tests/unit/test_simulation.py -v     # Simulation tests only (37)
uv run pytest tests/ --cov=src/agentique           # With coverage
```

### Key Packages

| Package | Purpose |
|---------|---------|
| `src/agentique/simulation/` | Simulation system (agent, harness, UI, MCP server) |
| `src/agentique/testing/client/` | Test client (MCPTestClient, scenarios, autonomous agent) |
| `src/agentique/core/` | Core abstractions (Agent, Task, Registry) |
| `src/agentique/bridge/` | Bridge layer (Router, Provider, FastMCP middleware) |
| `src/agentique/adapters/` | Protocol adapters (A2A) |

### Docker Compose Files

| File | Purpose |
|------|---------|
| `docker-compose.simulation.yml` | Simulation stack (4 services: agent, mcp, sim-ui, sim-mcp) |
| `docker-compose.interactive.yml` | Interactive testing stack (3 services: agent, mcp, test-ui) |

### Project Layout

```
src/agentique/
├── simulation/               # AI-powered human conversation simulation
│   ├── models.py             # SimulationPersona, Config, Result, EventType
│   ├── simulation_agent.py   # LLM-powered human emulator
│   ├── app.py                # SimulationHarness + REST API + WebSocket
│   ├── mcp_server/           # Simulator MCP Server (6 tools)
│   └── static/               # Observer UI (HTML/JS/CSS)
├── testing/client/           # Test client (MCPTestClient, scenarios)
├── core/                     # Agent, Task, Registry abstractions
├── bridge/                   # Router, Provider, FastMCP integration
├── adapters/a2a/             # A2A protocol adapter
└── server.py                 # Main Agentique MCP server
```

## Connection Guide

See [MCP_CONNECTION_GUIDE.md](./MCP_CONNECTION_GUIDE.md) for setup details.

## Resources

- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)
- [FastMCP Documentation](https://gofastmcp.com/)
- [A2A Protocol](https://github.com/google/agent-to-agent-protocol)
