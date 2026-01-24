# AgentMCP Testing Guide

Complete testing guide for the AgentMCP bridge, from individual components to end-to-end integration.

---

## Prerequisites

```bash
# Set your Google API key (REQUIRED)
export GOOGLE_API_KEY=your_actual_api_key_here

# Navigate to the project directory
cd /home/cairon/git/AgentMCP
```

---

## Layer 1: Test the ADK Agent (Python)

### Option A: Test Locally (without Docker)

```bash
# Install the test agent
cd a2a_test_agent
uv pip install -e .

# Configure environment
export A2A_HOST=127.0.0.1
export A2A_PORT=9000
export A2A_PROTOCOL=http
export A2A_BASE_URL=http://127.0.0.1:9000
export GOOGLE_API_KEY=your_key_here
export ADK_MODEL=gemini-2.0-flash

# Start the agent
uv run adk-test-agent
```

Expected output:
```
INFO - Starting A2A server on 127.0.0.1:9000
INFO - Building root agent...
INFO - Root agent 'TestAgentRoot' built successfully
INFO - A2A application created successfully
INFO - Uvicorn running on http://127.0.0.1:9000
```

### Option B: Test with Docker

```bash
# From AgentMCP root directory
cd /home/cairon/git/AgentMCP

# Build and start just the A2A agent
docker compose up --build a2a-test-agent
```

---

## Layer 2: Test the A2A Server

### 2.1: Verify Agent Card (Health Check)

```bash
# Test agent card endpoint
curl http://localhost:9000/.well-known/agent-card.json | jq
```

Expected output (condensed):
```json
{
  "name": "TestAgentRoot",
  "description": "Multi-capability test agent...",
  "url": "http://...",
  "version": "...",
  "capabilities": {...},
  "skills": [...]
}
```

### 2.2: Test Simple Calculator Request

```bash
curl -X POST http://localhost:9000/a2a/TestAgentRoot \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "parts": [{"text": "What is 25 + 17?"}],
      "message_id": "test-1"
    }]
  }' | jq
```

Expected: Response with calculation result (42)

### 2.3: Test Statistical Analysis

```bash
curl -X POST http://localhost:9000/a2a/TestAgentRoot \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "parts": [{"text": "Calculate statistics for these numbers: 5, 10, 15, 20, 25"}],
      "message_id": "test-2"
    }]
  }' | jq
```

Expected: Statistical results (mean=15, median=15, min=5, max=25, sum=75)

### 2.4: Test Data Processing

```bash
curl -X POST http://localhost:9000/a2a/TestAgentRoot \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "parts": [{"text": "Sort these words alphabetically: zebra, apple, mango, banana"}],
      "message_id": "test-3"
    }]
  }' | jq
```

Expected: Sorted list ["apple", "banana", "mango", "zebra"]

### 2.5: Test Text Processing

```bash
curl -X POST http://localhost:9000/a2a/TestAgentRoot \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "parts": [{"text": "Convert hello world to title case"}],
      "message_id": "test-4"
    }]
  }' | jq
```

Expected: "Hello World"

### 2.6: Test Information Retrieval

```bash
curl -X POST http://localhost:9000/a2a/TestAgentRoot \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "parts": [{"text": "Tell me about Python"}],
      "message_id": "test-5"
    }]
  }' | jq
```

Expected: Information about Python programming language from mock knowledge base

### 2.7: Test Multi-Step Workflow

```bash
curl -X POST http://localhost:9000/a2a/TestAgentRoot \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "parts": [{"text": "Calculate 10 + 20 + 30, then tell me a random fact"}],
      "message_id": "test-6"
    }]
  }' | jq
```

Expected: Sum (60) followed by a random fact

---

## Layer 3: Test the MCP Server

### 3.1: Start the MCP Server

```bash
# In a new terminal, from AgentMCP root
cd /home/cairon/git/AgentMCP

# Install agentique
uv pip install -e '.[a2a]'

# Configure environment
export AGENTIQUE_TRANSPORT=http
export AGENTIQUE_HOST=127.0.0.1
export AGENTIQUE_PORT=8000
export AGENTIQUE_AGENTS="root=http://localhost:9000|calculator,data_processing,text_manipulation,info_retrieval"

# Start MCP server
uv run python -m agentique
```

Expected output:
```
FastMCP server starting...
Server running on http://127.0.0.1:8000
```

### 3.2: Test MCP Tool - List Agents

```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_list_agents",
    "arguments": {}
  }' | jq
```

Expected: List of registered agents with their skills

### 3.3: Test MCP Tool - Send Message

```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "What is 15 times 3?",
      "agent": "root"
    }
  }' | jq
```

Expected: Response with calculation result (45)

### 3.4: Test MCP Tool - Send with Skill Routing

```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Count words in: The quick brown fox jumps over the lazy dog",
      "skill": "text_manipulation"
    }
  }' | jq
```

Expected: Word count (9 words)

### 3.5: Test MCP Resource - Agent List

```bash
curl http://localhost:8000/resources/a2a://agents | jq
```

Expected: Catalog of all agents

### 3.6: Test MCP Resource - Agent Details

```bash
curl http://localhost:8000/resources/a2a://agents/root | jq
```

Expected: Details about the root agent

### 3.7: Test MCP Resource - Agent Card

```bash
curl http://localhost:8000/resources/a2a://agents/root/card | jq
```

Expected: Full A2A agent card

### 3.8: Test MCP Prompt

```bash
curl -X POST http://localhost:8000/prompts/get \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_routing_prompt",
    "arguments": {
      "goal": "Calculate statistics for my data"
    }
  }' | jq
```

Expected: Routing suggestions for the goal

---

## Layer 4: Full Docker Compose Setup

### 4.1: Start All Services

```bash
cd /home/cairon/git/AgentMCP

# Start both services
export GOOGLE_API_KEY=your_key_here
docker compose up --build
```

### 4.2: Verify Both Services Are Running

```bash
# Check A2A agent
curl http://localhost:9000/.well-known/agent-card.json | jq '.name'

# Check MCP server (list agents)
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "a2a_list_agents", "arguments": {}}' | jq
```

### 4.3: Test End-to-End Flow

```bash
# Send a message through MCP to A2A agent
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Calculate the mean of these numbers: 10, 20, 30, 40, 50",
      "agent": "root"
    }
  }' | jq '.result.text'
```

Expected: Agent calculates mean (30)

---

## Layer 5: Connect MCP Client

### 5.1: Using Claude Desktop (If Available)

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "agentique": {
      "command": "uv",
      "args": ["run", "python", "-m", "agentique"],
      "cwd": "/home/cairon/git/AgentMCP",
      "env": {
        "AGENTIQUE_AGENTS": "root=http://localhost:9000|calculator,data_processing,text_manipulation,info_retrieval"
      }
    }
  }
}
```

Restart Claude Desktop and the tools should appear.

### 5.2: Using MCP Inspector (Development Tool)

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run inspector
mcp-inspector uv run python -m agentique
```

This opens a web UI where you can:
- See all available tools, resources, and prompts
- Test tools interactively
- View request/response logs

### 5.3: Using Python MCP Client

```python
# test_mcp_client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "agentique"],
        env={
            "AGENTIQUE_AGENTS": "root=http://localhost:9000|calculator,data_processing,text_manipulation,info_retrieval"
        }
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")

            # Call a2a_send
            result = await session.call_tool("a2a_send", {
                "message": "What is 100 divided by 5?",
                "agent": "root"
            })
            print(f"Result: {result}")

asyncio.run(test_mcp())
```

---

## Layer 6: Test Prompts & Functionality

### Test Suite: Comprehensive Agent Testing

#### Test 1: Basic Arithmetic
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "What is 456 + 789?",
      "skill": "calculator"
    }
  }' | jq '.result.text'
```
**Expected**: 1245

#### Test 2: Statistical Analysis
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Calculate statistics for: 3, 7, 8, 5, 12, 14, 21, 13, 18",
      "skill": "calculator"
    }
  }' | jq '.result.text'
```
**Expected**: count=9, mean≈11.2, median=12, min=3, max=21, sum=101

#### Test 3: List Filtering
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Filter this list by items containing the letter a: apple, banana, orange, grape, kiwi, pear",
      "skill": "data_processing"
    }
  }' | jq '.result.text'
```
**Expected**: apple, banana, orange, grape, pear

#### Test 4: Sorting
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Sort these in reverse alphabetical order: dog, cat, bird, fish, hamster",
      "skill": "data_processing"
    }
  }' | jq '.result.text'
```
**Expected**: hamster, fish, dog, cat, bird

#### Test 5: Word Count
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Count the words in this sentence: Machine learning is transforming how we build intelligent systems",
      "skill": "text_manipulation"
    }
  }' | jq '.result.text'
```
**Expected**: 10 words

#### Test 6: Case Transformation
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Convert this to uppercase: hello world from agents",
      "skill": "text_manipulation"
    }
  }' | jq '.result.text'
```
**Expected**: HELLO WORLD FROM AGENTS

#### Test 7: Keyword Extraction
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Extract keywords from: Artificial intelligence and machine learning are revolutionizing technology. Deep learning models continue to advance rapidly.",
      "skill": "text_manipulation"
    }
  }' | jq '.result.text'
```
**Expected**: Top 5 keywords (e.g., intelligence, machine, learning, technology, models)

#### Test 8: Information Retrieval
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "What do you know about MCP?",
      "skill": "info_retrieval"
    }
  }' | jq '.result.text'
```
**Expected**: Information about Model Context Protocol

#### Test 9: Random Fact
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Tell me a random fact",
      "skill": "info_retrieval"
    }
  }' | jq '.result.text'
```
**Expected**: One of the predefined random facts

#### Test 10: Complex Multi-Agent Workflow
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "First, calculate the sum of 10, 20, and 30. Then convert that result to uppercase text.",
      "agent": "root"
    }
  }' | jq '.result.text'
```
**Expected**: Agent routes to Calculator (sum=60), then to TextProcessor (SIXTY or formatted result)

#### Test 11: Streaming Response
```bash
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_stream",
    "arguments": {
      "message": "Calculate statistics for 1 through 100",
      "skill": "calculator"
    }
  }' | jq -c '.result[]'
```
**Expected**: Stream of chunks showing incremental progress

#### Test 12: Agent Routing by Skill
```bash
# Math skill
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Multiply 15 by 8",
      "skill": "calculator"
    }
  }' | jq '.result.agent'

# Text skill
curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "Reverse this text: hello",
      "skill": "text_manipulation"
    }
  }' | jq '.result.agent'
```
**Expected**: Both should route to "root" agent, which then delegates internally

---

## Debugging Tips

### Check Logs

```bash
# Docker logs
docker compose logs a2a-test-agent
docker compose logs mcp-server

# Follow logs in real-time
docker compose logs -f
```

### Common Issues

#### 1. Agent Not Responding

```bash
# Check if agent is healthy
docker ps
docker inspect agentmcp-a2a-test-agent | jq '.[0].State.Health'
```

#### 2. MCP Server Can't Connect

```bash
# Test connectivity from MCP container to A2A container
docker exec agentmcp-mcp-server curl http://a2a-test-agent:9000/.well-known/agent-card.json
```

#### 3. API Key Issues

```bash
# Verify API key is set in container
docker exec agentmcp-a2a-test-agent env | grep GOOGLE_API_KEY
```

### Test Environment Variables

```bash
# Check what the agent sees
docker exec agentmcp-a2a-test-agent env | grep A2A

# Check what MCP server sees
docker exec agentmcp-mcp-server env | grep AGENTIQUE
```

---

## Performance Testing

### Measure Response Time

```bash
# Test latency
time curl -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "a2a_send",
    "arguments": {
      "message": "What is 2 + 2?",
      "agent": "root"
    }
  }' | jq '.result.text'
```

### Concurrent Requests

```bash
# Test with multiple concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:8000/tools/call \
    -H "Content-Type: application/json" \
    -d "{
      \"name\": \"a2a_send\",
      \"arguments\": {
        \"message\": \"Calculate $i times 10\",
        \"agent\": \"root\"
      }
    }" &
done
wait
```

---

## Success Criteria

✅ **Layer 1**: Agent starts without errors
✅ **Layer 2**: Agent card accessible, A2A messages get responses
✅ **Layer 3**: MCP server lists agents, tools work correctly
✅ **Layer 4**: Both services communicate through Docker network
✅ **Layer 5**: MCP client can discover and use tools
✅ **Layer 6**: All test prompts return expected results

---

## Quick Test Script

Save this as `test_all.sh`:

```bash
#!/bin/bash
set -e

echo "🧪 AgentMCP Test Suite"
echo "====================="
echo ""

echo "✓ Testing A2A Agent Card..."
curl -s http://localhost:9000/.well-known/agent-card.json | jq -e '.name' > /dev/null && echo "  ✅ Agent card OK" || echo "  ❌ Agent card FAILED"

echo "✓ Testing A2A Message..."
curl -s -X POST http://localhost:9000/a2a/TestAgentRoot \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "parts": [{"text": "What is 5 + 3?"}], "message_id": "test"}]}' \
  | jq -e '.messages' > /dev/null && echo "  ✅ A2A messaging OK" || echo "  ❌ A2A messaging FAILED"

echo "✓ Testing MCP List Agents..."
curl -s -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "a2a_list_agents", "arguments": {}}' \
  | jq -e '.result' > /dev/null && echo "  ✅ MCP list agents OK" || echo "  ❌ MCP list agents FAILED"

echo "✓ Testing MCP Send Message..."
curl -s -X POST http://localhost:8000/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "a2a_send", "arguments": {"message": "What is 10 + 5?", "agent": "root"}}' \
  | jq -e '.result.text' > /dev/null && echo "  ✅ MCP send OK" || echo "  ❌ MCP send FAILED"

echo ""
echo "====================="
echo "🎉 Test suite complete!"
```

Run with:
```bash
chmod +x test_all.sh
./test_all.sh
```

---

## Next Steps

After successful testing:

1. **Explore Agent Capabilities**: Try different combinations of subagents
2. **Test Context Preservation**: Send multi-turn conversations
3. **Benchmark Performance**: Measure latency and throughput
4. **Extend Agents**: Add new subagents with different capabilities
5. **Build Applications**: Use the MCP bridge in real applications

Happy testing! 🚀
