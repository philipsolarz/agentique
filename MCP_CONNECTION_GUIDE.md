# Connecting Claude Code to Agentique

This guide shows you how to connect Claude Code to Agentique's MCP servers for **testing** and **simulation**.

## Two MCP Servers

Agentique provides two complementary MCP servers:

| Server | Purpose | Module |
|--------|---------|--------|
| **Test Suite** | Run autonomous tests, scenarios, get bug reports | `agentique.testing.client.mcp_server` |
| **Simulator** | Start simulated human conversations, observe results | `agentique.simulation.mcp_server` |

## Quick Start — Simulator (Recommended)

The Simulator creates AI-powered human personas that have natural conversations with your agents through the full MCP pipeline, discovering bugs and UX issues.

### 1. Start the Docker Stack

```bash
echo "GOOGLE_API_KEY=your-key-here" > .env
docker compose -f docker-compose.simulation.yml up -d --wait
```

### 2. Configure Claude Code

Add to `.claude/mcp.json` in the project:

```json
{
  "mcpServers": {
    "agentique-simulator": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "agentique.simulation.mcp_server",
        "--simulation-ui-url",
        "http://localhost:8080",
        "--mcp-url",
        "http://localhost:8000/mcp"
      ],
      "cwd": "/home/cairon/git/AgentMCP",
      "env": {
        "GOOGLE_API_KEY": "${GOOGLE_API_KEY}"
      }
    }
  }
}
```

### 3. Restart Claude Code and Use

```
Start a simulation with a curious developer exploring math capabilities
```

### 4. Watch in Browser

Open http://localhost:8080 to see the simulated conversation in real-time with typing animations.

## Simulator MCP Tools

### `start_simulation`
Start a simulated human conversation.
- `objective` (string): What the simulated person should explore
- `persona_name` (string): Name of the simulated person (default: "Alex")
- `persona_role` (string): Background (e.g., "curious developer")
- `conversation_style` (string): How they talk (e.g., "casual and direct")
- `max_turns` (int): Max conversation turns (default: 10)
- `topics` (list[str]): Specific topics to explore

### `stop_simulation`
Stop a running simulation.
- `simulation_id` (string): ID from start_simulation

### `get_simulation_status`
Get current state, turns completed, insights count.
- `simulation_id` (string)

### `get_simulation_events`
Get detailed event log with pagination.
- `simulation_id` (string)
- `limit` (int, default: 50)
- `offset` (int, default: 0)

### `get_simulation_insights`
Get bugs, UX issues, and observations discovered during simulation.
- `simulation_id` (string)

### `list_simulations`
List all active and completed simulations.

## Headless Mode

The Simulator MCP Server can run without the UI (headless mode). Omit `--simulation-ui-url`:

```json
{
  "mcpServers": {
    "agentique-simulator": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "agentique.simulation.mcp_server",
        "--mcp-url",
        "http://localhost:8000/mcp"
      ],
      "cwd": "/home/cairon/git/AgentMCP",
      "env": {
        "GOOGLE_API_KEY": "${GOOGLE_API_KEY}"
      }
    }
  }
}
```

## Test Suite Setup (Original)

The Test Suite MCP Server runs autonomous tests and predefined scenarios.

### Configuration

```json
{
  "mcpServers": {
    "agentique-test-suite": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "agentique.testing.client.mcp_server",
        "--mcp-url",
        "http://localhost:8000/mcp"
      ],
      "cwd": "/home/cairon/git/AgentMCP",
      "env": {
        "GOOGLE_API_KEY": "${GOOGLE_API_KEY}"
      }
    }
  }
}
```

### Test Suite Tools

- `run_autonomous_test` — AI-powered testing with bug discovery
- `get_last_test_report` — Most recent test report
- `get_test_insights` — All bugs categorized by severity
- `list_mcp_tools` — Tools available on the MCP server
- `run_test_scenario` — Run a specific YAML scenario
- `list_test_scenarios` — List all scenarios

## Architecture

```
Claude Code ──stdio──> Simulator MCP Server (local)
                              │
                              │ REST API
                              v
                       Simulation UI (Docker :8080)
                              │
                              │ MCPTestClient
                              v
                       MCP Server (Docker :8000)
                              │
                              │ A2A Protocol
                              v
                       A2A Agent (Docker :9000)
```

The Simulator MCP Server runs locally via stdio, while the rest of the stack runs in Docker. This means the MCP connection survives Docker rebuilds.

## Troubleshooting

### Tools Not Available

```bash
uv --version                    # Check uv is installed
docker compose -f docker-compose.simulation.yml ps  # Check Docker stack
curl http://localhost:8000/health                     # Check MCP server
curl http://localhost:8080/api/simulations            # Check simulation UI
```

### Restart After Code Changes

```bash
docker compose -f docker-compose.simulation.yml up -d --build
# Claude Code MCP connection survives — no restart needed
```
