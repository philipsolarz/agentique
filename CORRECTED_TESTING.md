# AgentMCP Corrected Testing Guide

**Important Discoveries:**
- The A2A server uses **JSON-RPC** protocol, not REST
- The endpoint is `POST /` with method `message/send`, not `POST /a2a/TestAgentRoot`
- FastMCP 3.0 uses the **MCP protocol over SSE**, not direct HTTP tool calls
- MCP server must be tested with an MCP client, not curl

---

## Layer 1: Test the A2A Agent (JSON-RPC)

### Check Agent Card (Health Check)

```bash
curl http://localhost:9000/.well-known/agent-card.json | jq '.name'
```

Expected: `"TestAgentRoot"`

### Test Simple Calculation (JSON-RPC Format)

```bash
curl -s -X POST http://localhost:9000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "What is 25 + 17?"}],
        "message_id": "test-1"
      }
    },
    "id": 1
  }' | jq '.result.artifacts[0].parts[0].text'
```

Expected: `"25 + 17 = 42"`

### Test Multi-Agent Routing

```bash
curl -s -X POST http://localhost:9000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Calculate statistics for: 5, 10, 15, 20, 25"}],
        "message_id": "test-stats"
      }
    },
    "id": 2
  }' | jq '.result.artifacts[0].parts[0].text'
```

Expected: Statistics results (mean=15, etc.)

### Test Text Processing

```bash
curl -s -X POST http://localhost:9000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Convert hello world to UPPERCASE"}],
        "message_id": "test-text"
      }
    },
    "id": 3
  }' | jq '.result.artifacts[0].parts[0].text'
```

Expected: `"HELLO WORLD"`

### Check Task Status and History

```bash
curl -s -X POST http://localhost:9000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "What is 10 + 20?"}],
        "message_id": "test-trace"
      }
    },
    "id": 4
  }' | jq '{
    status: .result.status.state,
    agent: .result.metadata.adk_author,
    artifact: .result.artifacts[0].parts[0].text,
    history_length: (.result.history | length)
  }'
```

Expected:
```json
{
  "status": "completed",
  "agent": "Calculator",
  "artifact": "10 + 20 = 30",
  "history_length": 7
}
```

---

## Layer 2: Test the MCP Server

The MCP server uses the MCP protocol over SSE and requires an MCP client to test.

### Method 1: Using Python MCP Client (Recommended)

Install dependencies:
```bash
pip install mcp
```

Run the test script:
```bash
cd /home/cairon/git/AgentMCP
python test_mcp_client.py
```

Expected output:
```
🧪 Test 1: List Agents
============================================================
✓ Available tools: ['a2a_send', 'a2a_stream', 'a2a_list_agents']
✓ Found 1 agent(s)
  - root: skills=['calculator', 'data_processing', 'text_manipulation', 'info_retrieval']

🧪 Test 2: Send Message (Calculator)
============================================================
✓ Agent: root
✓ Response: 100 divided by 5 is 20
✓ Events: X event(s)

🧪 Test 3: Send Message by Skill (Text Manipulation)
============================================================
✓ Agent: root
✓ Response: Hello World

============================================================
✅ All tests passed!
============================================================
```

### Method 2: Using MCP Inspector (Development UI)

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run inspector (from AgentMCP directory)
cd /home/cairon/git/AgentMCP
mcp-inspector python -m agentique
```

This opens a web UI where you can:
- See all tools, resources, and prompts
- Test tools interactively
- View request/response logs in real-time

### Method 3: Using Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "agentique": {
      "command": "python",
      "args": ["-m", "agentique"],
      "env": {
        "AGENTIQUE_AGENTS": "root=http://localhost:9000|calculator,data_processing,text_manipulation,info_retrieval"
      }
    }
  }
}
```

Restart Claude Desktop and tools will appear in the interface.

---

## Complete Test Prompts

### Test 1: Basic Arithmetic
**Prompt**: "What is 456 + 789?"
**Expected**: 1245
**Validates**: Calculator agent, add function

### Test 2: Statistical Analysis
**Prompt**: "Calculate statistics for these numbers: 3, 7, 8, 5, 12, 14, 21, 13, 18"
**Expected**: count=9, mean≈11.2, median=12, min=3, max=21
**Validates**: Calculator agent, calculate_statistics function

### Test 3: Division
**Prompt**: "What is 100 divided by 5?"
**Expected**: 20
**Validates**: Calculator agent, divide function

### Test 4: List Sorting
**Prompt**: "Sort these in reverse order: dog, cat, bird, fish"
**Expected**: fish, dog, cat, bird
**Validates**: DataProcessor agent, sort_list function

### Test 5: List Filtering
**Prompt**: "Filter items containing 'a': apple, banana, orange, grape"
**Expected**: apple, banana, orange, grape
**Validates**: DataProcessor agent, filter_list function

### Test 6: Word Count
**Prompt**: "Count words in: The quick brown fox jumps"
**Expected**: 5 words
**Validates**: TextProcessor agent, count_words function

### Test 7: Case Conversion
**Prompt**: "Convert to uppercase: hello world"
**Expected**: HELLO WORLD
**Validates**: TextProcessor agent, transform_case function

### Test 8: Text Reversal
**Prompt**: "Reverse: hello"
**Expected**: olleh
**Validates**: TextProcessor agent, reverse_text function

### Test 9: Knowledge Lookup
**Prompt**: "What do you know about Python?"
**Expected**: Information about Python programming language
**Validates**: InfoRetriever agent, search_info function

### Test 10: Random Fact
**Prompt**: "Tell me a random fact"
**Expected**: One of the predefined facts
**Validates**: InfoRetriever agent, get_random_fact function

### Test 11: Multi-Step Workflow
**Prompt**: "Calculate 10 + 20 + 30, then reverse that number as text"
**Expected**: Agent coordinates Calculator → result (60) → TextProcessor
**Validates**: Multi-agent orchestration

### Test 12: Skill-Based Routing
Send same message with different skills:
- **Skill: calculator** → Routes to Calculator
- **Skill: text_manipulation** → Routes to TextProcessor
**Validates**: Skill-based agent resolution

---

## Quick Reference

### A2A JSON-RPC Format

```json
{
  "jsonrpc": "2.0",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"text": "Your message here"}],
      "message_id": "unique-id"
    }
  },
  "id": 1
}
```

### Extract Response Text

```bash
jq '.result.artifacts[0].parts[0].text'
```

### Extract Task Status

```bash
jq '.result.status.state'
```

### Extract Agent That Handled Request

```bash
jq '.result.metadata.adk_author'
```

---

## Debugging

### Check Logs

```bash
# A2A agent logs
docker logs agentmcp-a2a-test-agent -f

# MCP server logs
docker logs agentmcp-mcp-server -f
```

### Verify Environment

```bash
# Check A2A agent environment
docker exec agentmcp-a2a-test-agent env | grep -E "(ADK_MODEL|GOOGLE_API)"

# Expected:
# GOOGLE_API_KEY=your_key
# ADK_MODEL=gemini-3-flash-preview
```

### Test A2A Connectivity from MCP Container

```bash
docker exec agentmcp-mcp-server curl http://a2a-test-agent:9000/.well-known/agent-card.json
```

---

## Success Criteria

✅ **A2A Agent**: Returns `"status": "completed"` in JSON-RPC responses
✅ **Multi-Agent Routing**: `metadata.adk_author` shows correct subagent name
✅ **MCP Server**: Python test script passes all tests
✅ **End-to-End**: Message → MCP → A2A → Agent → Response

---

## Key Differences from Original Testing Guide

1. **A2A Endpoint**: Use `POST /` with JSON-RPC, not `POST /a2a/TestAgentRoot`
2. **A2A Method**: Use `"method": "message/send"` in JSON-RPC request
3. **Response Format**: Result is in `.result.artifacts[0].parts[0].text`, not `.messages`
4. **MCP Testing**: Use MCP client (Python, Inspector, Claude Desktop), not HTTP curl
5. **Model Name**: Use `gemini-3-flash-preview` or `gemini-2.0-flash`, not `gemini-3.0-flash`

---

## Next Steps

1. ✅ Verify A2A agent works with JSON-RPC format
2. ✅ Run Python MCP client tests
3. ⏭️ Try MCP Inspector for interactive testing
4. ⏭️ Integrate with Claude Desktop
5. ⏭️ Build your own MCP client applications

Happy testing! 🚀
