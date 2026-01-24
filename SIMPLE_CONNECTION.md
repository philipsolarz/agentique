# Simple HTTP Connection Guide

Your AgentMCP server is accessible via HTTP at:

## 🌐 Connection URL

```
http://localhost:8000/mcp
```

That's it! Any MCP client that supports HTTP transport can connect to this URL.

---

## Quick Test

### Python FastMCP Client

```bash
cd /home/cairon/git/AgentMCP
uv run python test_http_client.py
```

### TypeScript/JavaScript Client

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

const transport = new SSEClientTransport(
  new URL("http://localhost:8000/mcp")
);

const client = new Client({
  name: "my-app",
  version: "1.0.0"
}, {
  capabilities: {}
});

await client.connect(transport);

// List tools
const tools = await client.listTools();
console.log(tools);

// Call a tool
const result = await client.callTool({
  name: "a2a_send",
  arguments: {
    message: "What is 25 + 17?",
    agent: "root"
  }
});
```

### curl (for debugging)

```bash
# The endpoint expects SSE/Streamable HTTP protocol
# Direct curl testing is complex, use the Python/JS clients above
```

---

## Available Tools

Once connected, you have access to:

### `a2a_send`
Send a message to an A2A agent and get a response.

**Parameters:**
- `message` (string, required): The message to send
- `agent` (string, optional): Agent name (defaults to "root")
- `skill` (string, optional): Route by skill instead of name

**Example:**
```python
result = await client.call_tool("a2a_send", {
    "message": "Calculate 10 + 20",
    "agent": "root"
})
```

### `a2a_stream`
Stream responses from an agent.

**Parameters:** Same as `a2a_send`

### `a2a_list_agents`
List all available agents and their skills.

**Parameters:** None

**Example:**
```python
agents = await client.call_tool("a2a_list_agents", {})
```

---

## Environment Setup

Make sure these services are running:

```bash
# Start A2A agent
docker compose up a2a-test-agent

# Start MCP server (in another terminal)
cd /home/cairon/git/AgentMCP
export AGENTIQUE_TRANSPORT=http
export AGENTIQUE_HOST=0.0.0.0
export AGENTIQUE_PORT=8000
export AGENTIQUE_AGENTS="root=http://localhost:9000|calculator,data_processing,text_manipulation,info_retrieval"
uv run python -m agentique
```

Or use Docker Compose for both:

```bash
docker compose up
```

---

## Web Application Integration

For web apps, you can connect directly to `http://localhost:8000/mcp`:

### React Example

```typescript
import { Client } from '@modelcontextprotocol/sdk/client';
import { StreamableHttpTransport } from '@modelcontextprotocol/sdk/client/http';

const transport = new StreamableHttpTransport({
  url: 'http://localhost:8000/mcp'
});

const client = new Client(transport);

// In your component
async function calculate() {
  const result = await client.callTool({
    name: 'a2a_send',
    arguments: {
      message: 'What is 100 / 5?',
      agent: 'root'
    }
  });

  return result.content[0].text;
}
```

### Next.js API Route

```typescript
// app/api/agent/route.ts
import { Client } from '@modelcontextprotocol/sdk/client';
import { StreamableHttpTransport } from '@modelcontextprotocol/sdk/client/http';

export async function POST(request: Request) {
  const { message } = await request.json();

  const transport = new StreamableHttpTransport({
    url: 'http://localhost:8000/mcp'
  });

  const client = new Client(transport);

  const result = await client.callTool({
    name: 'a2a_send',
    arguments: { message, agent: 'root' }
  });

  return Response.json({
    answer: result.content[0].text
  });
}
```

---

## Production Deployment

For production, deploy both services and update the URL:

1. **Deploy A2A Agent** (e.g., Cloud Run, Railway, Fly.io)
   ```bash
   # Will be available at: https://a2a-agent.your-domain.com
   ```

2. **Update MCP Server config**
   ```bash
   export AGENTIQUE_AGENTS="root=https://a2a-agent.your-domain.com|calculator,data_processing,text_manipulation,info_retrieval"
   ```

3. **Deploy MCP Server** (e.g., Cloud Run, Railway, Fly.io)
   ```bash
   # Will be available at: https://mcp-server.your-domain.com/mcp
   ```

4. **Connect your app**
   ```typescript
   const transport = new StreamableHttpTransport({
     url: 'https://mcp-server.your-domain.com/mcp'
   });
   ```

---

## Troubleshooting

### Connection refused

Make sure the MCP server is running:
```bash
curl http://localhost:8000/mcp
# Should NOT return "Connection refused"
```

### "Not Acceptable" error

The client needs to support SSE/Streamable HTTP protocol. Use the FastMCP client or MCP SDK clients.

### Tools not showing up

Check server logs:
```bash
docker logs agentmcp-mcp-server
# Should show tools registered
```

---

## Summary

**Connection URL:** `http://localhost:8000/mcp`

**That's all you need!** Just point any MCP-compatible HTTP client to that URL and you're connected.

No complex configuration, no STDIO setup, just a simple HTTP endpoint.
