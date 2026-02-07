# MCP Client Configuration Examples

This directory contains configuration examples for connecting popular MCP clients to the agentique bridge server.

## Prerequisites

1. Start the interactive Docker stack:
   ```bash
   docker compose -f docker-compose.interactive.yml up -d --wait
   ```

2. Verify the services are running:
   ```bash
   docker ps | grep agentique-interactive
   ```

## Claude CLI

**Location:** `~/.claude/mcp_settings.json`

```json
{
  "mcpServers": {
    "agentique": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "agentique-interactive-mcp-server",
        "python",
        "-m",
        "agentique.server",
        "--stdio"
      ]
    }
  }
}
```

**Test:**
```bash
claude mcp list
claude "Tell me about the available agents"
```

## Claude Desktop

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`

Copy the contents of `claude-desktop-config.json` to your Claude Desktop configuration file.

**Test:**
1. Restart Claude Desktop
2. Look for "agentique-bridge" in the MCP servers list
3. Try: "Show me available tools" or "Call the agents tool"

## VS Code (Cline / Continue / GitHub Copilot)

**Location:** `.vscode/settings.json` (workspace) or VS Code User Settings

For Cline:
```json
{
  "cline.mcpServers": {
    "agentique": {
      "command": "docker",
      "args": [
        "exec", "-i",
        "agentique-interactive-mcp-server",
        "python", "-m", "agentique.server", "--stdio"
      ]
    }
  }
}
```

For Continue:
```json
{
  "continue.mcpServers": [{
    "name": "agentique",
    "command": "docker",
    "args": [
      "exec", "-i",
      "agentique-interactive-mcp-server",
      "python", "-m", "agentique.server", "--stdio"
    ]
  }]
}
```

## HTTP Mode (for web-based clients)

If your MCP client connects via HTTP instead of stdio:

```
http://localhost:8000/mcp
```

Configure the Docker Compose stack to expose the HTTP endpoint (already done in docker-compose.interactive.yml).

## Alternative: Direct Python Execution

If you prefer not to use Docker:

```json
{
  "mcpServers": {
    "agentique": {
      "command": "python",
      "args": [
        "-m",
        "agentique.server",
        "--stdio"
      ],
      "env": {
        "A2A_AGENT_URL": "http://localhost:9000",
        "PYTHONPATH": "/path/to/AgentMCP/src"
      }
    }
  }
}
```

**Note:** Make sure to start the A2A demo agent separately:
```bash
docker compose -f docker-compose.interactive.yml up -d a2a-demo-agent
```

## Troubleshooting

### Server not responding

1. Check Docker containers are running:
   ```bash
   docker ps | grep agentique-interactive
   ```

2. Check server logs:
   ```bash
   docker logs agentique-interactive-mcp-server
   ```

3. Test server directly:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | docker exec -i agentique-interactive-mcp-server python -m agentique.server --stdio
   ```

### Connection refused

- Ensure the Docker stack is running (`docker compose up`)
- Check port 8000 is not in use: `lsof -i :8000`
- Verify network connectivity: `curl http://localhost:8000/health`

### Tools not showing up

- Check MCP client logs for errors
- Verify the server exposes tools: `docker logs agentique-interactive-mcp-server | grep tool`
- Try restarting the MCP client

## Verification Checklist

After connecting an MCP client:

- [ ] Client recognizes the agentique server
- [ ] Tools are listed (agents, agent, task, inspect, agent_background)
- [ ] Can call the `agents` tool to list available agents
- [ ] Can send a message to the demo agent
- [ ] Streaming responses work
- [ ] Multi-turn conversations maintain context
- [ ] Error handling works (try `/error` command)

See `INTERACTIVE_TESTING.md` for detailed test scenarios.
