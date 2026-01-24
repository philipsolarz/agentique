# AgentMCP Quick Start

Fast-track guide to get AgentMCP running in under 5 minutes.

## Step 1: Set API Key

```bash
export GOOGLE_API_KEY=your_actual_api_key_here
```

Get your key from: https://makersuite.google.com/app/apikey

## Step 2: Start Services

```bash
cd /home/cairon/git/AgentMCP
docker compose up --build
```

Wait for:
```
agentmcp-a2a-test-agent  | INFO:     Uvicorn running on http://0.0.0.0:9000
agentmcp-mcp-server      | FastMCP server running...
```

## Step 3: Test

```bash
# In a new terminal, run the test suite
./test_all.sh
```

Expected output:
```
🧪 AgentMCP Test Suite
=====================

✓ Testing A2A Agent Card...
  ✅ Agent card OK
✓ Testing A2A Message...
  ✅ A2A messaging OK
✓ Testing MCP List Agents...
  ✅ MCP list agents OK
✓ Testing MCP Send Message...
  ✅ MCP send OK

=====================
🎉 Test suite complete!
```

## Quick Tests

### Test 1: Simple Math
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "What is 25 + 17?",
      "agent": "root"
    }
  }' | jq '.result.text'
```

### Test 2: Statistics
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Calculate statistics for: 5, 10, 15, 20, 25",
      "skill": "calculator"
    }
  }' | jq '.result.text'
```

### Test 3: Text Processing
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Convert hello world to title case",
      "skill": "text_manipulation"
    }
  }' | jq '.result.text'
```

## URLs

- **A2A Agent**: http://localhost:9000
- **A2A Agent Card**: http://localhost:9000/.well-known/agent-card.json
- **MCP Server**: http://localhost:8000

## Tools Available

- `a2a_send` - Send a message to an agent
- `a2a_stream` - Stream agent responses
- `a2a_list_agents` - List all agents

## Resources Available

- `a2a://agents` - All agents
- `a2a://agents/{agent}` - Specific agent
- `a2a://agents/{agent}/card` - Agent card

## Skills Available

- `calculator` - Math and statistics
- `data_processing` - List manipulation
- `text_manipulation` - Text transformations
- `info_retrieval` - Information lookup

## Troubleshooting

**Services won't start?**
```bash
# Check logs
docker compose logs
```

**Can't connect?**
```bash
# Test connectivity
curl http://localhost:9000/.well-known/agent-card.json
```

**Agent not responding?**
```bash
# Check API key is set
docker exec agentmcp-a2a-test-agent env | grep GOOGLE_API_KEY
```

## Next Steps

See [TESTING.md](TESTING.md) for comprehensive testing guide with all test prompts.

## Stop Services

```bash
docker compose down
```
