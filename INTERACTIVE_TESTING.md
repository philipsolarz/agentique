# Interactive Testing Guide

This guide walks you through manual testing of agentique using real MCP clients (Claude CLI, GitHub Copilot, VS Code extensions, etc.).

## Prerequisites

- Docker and Docker Compose installed
- `GOOGLE_API_KEY` set in `.env` or environment (required for the Gemini-powered agent)

## Why Interactive Testing?

Automated tests validate protocol conformance, but interactive testing with real MCP clients ensures:
- Real-world MCP client compatibility
- User experience quality
- Streaming behavior in production
- Multi-turn conversation flows
- Error handling from a user perspective
- Tool discovery and invocation UX

## Quick Start

### 1. Start the Interactive Stack

```bash
# From the agentique repository root
./start-interactive.sh

# Or manually:
docker compose -f docker-compose.interactive.yml up -d --wait
```

Verify services are healthy:
```bash
# Check MCP server
curl http://localhost:8000/health

# Check A2A agent and its capabilities
curl http://localhost:9000/.well-known/agent-card.json
```

### 2. Configure Your MCP Client

Choose your preferred MCP client and follow the configuration guide in `examples/mcp-clients/README.md`:

- **Claude CLI**: `~/.claude/mcp_settings.json`
- **Claude Desktop**: Platform-specific config file
- **VS Code (Cline/Continue)**: `.vscode/settings.json`

### 3. Run Test Scenarios

Follow the scenarios below to systematically test all features.

## About the A2A Agent

The interactive testing stack runs a **real LLM-powered agent** built with Google ADK and Gemini. It features:

- **Natural language understanding** - no slash commands needed, just talk naturally
- **7 specialized sub-agents** routed by intent:
  - **Calculator** - arithmetic and statistics
  - **DataProcessor** - filter, sort, count lists
  - **TextProcessor** - case conversion, word count, keyword extraction
  - **InfoRetriever** - knowledge lookup and facts
  - **Interactive** - user confirmations, preferences, wizards
  - **Workflow** - background tasks, batch processing, state machines
  - **BranchDemo** - sub-agent visibility and hierarchy
- **25+ tool functions** with proper schemas
- **Streaming responses** with progress updates
- **Elicitation flows** for human-in-the-loop workflows

## Test Scenarios

### Scenario 1: Tool Discovery

**Objective**: Verify MCP client can discover tools exposed by the bridge.

**Steps**:
1. In your MCP client, list available tools
2. Verify you see: `agents`, `agent`, `task`, `inspect`, `agent_background`

**Expected Result**:
- All 5 core agentique tools appear
- Tool descriptions are clear
- Parameters are documented

**Claude CLI Example**:
```bash
claude mcp list
```

**In Chat**:
```
Show me the available MCP tools
```

### Scenario 2: List Available Agents

**Objective**: Test the `agents` tool that lists available A2A agents.

**Steps**:
1. Call the `agents` tool (no parameters)
2. Verify the agent appears with its skills

**Expected Result**:
- Response includes "TestAgentRoot"
- 7 skills listed (calculation, data_processing, text_manipulation, etc.)
- Agent capabilities shown (streaming, state history)

**In Chat**:
```
List all available agents
```

### Scenario 3: Math & Calculations

**Objective**: Test natural language routing to the Calculator sub-agent.

**Steps**:
1. Ask: "What is 15 times 7?"
2. Ask: "Calculate the statistics for 10, 20, 30, 40, 50"

**Expected Result**:
- Agent understands intent and routes to Calculator
- Returns correct results (105, mean/median/etc.)
- Uses the `calculate` tool internally

**In Chat**:
```
Ask the agent: What is 15 times 7?
```

### Scenario 4: Text Processing

**Objective**: Test routing to the TextProcessor sub-agent.

**Steps**:
1. Ask: "Convert 'hello world' to uppercase"
2. Ask: "Count the words in: The quick brown fox jumps over the lazy dog"
3. Ask: "Extract keywords from: Machine learning is transforming artificial intelligence"

**Expected Result**:
- Text transformations are correct
- Word count is accurate
- Keywords are relevant

**In Chat**:
```
Tell the agent: Convert 'hello world' to uppercase
```

### Scenario 5: Data Processing

**Objective**: Test routing to the DataProcessor sub-agent.

**Steps**:
1. Ask: "Sort these items: banana, apple, cherry, date"
2. Ask: "Filter items containing 'a' from: cat, dog, bat, rat, pig"

**Expected Result**:
- Items sorted alphabetically
- Filtering works correctly with patterns

**In Chat**:
```
Ask the agent to sort these items: banana, apple, cherry, date
```

### Scenario 6: Streaming Responses

**Objective**: Test streaming delivery through the bridge.

**Steps**:
1. Ask a question that triggers a longer response
2. Observe if response arrives in chunks

**Expected Result**:
- Response streams incrementally
- All chunks arrive
- Final response is complete

**In Chat**:
```
Ask the agent to process a large dataset of [1, 5, 3, 8, 2, 9, 4, 7, 6, 10] with sorting
```

### Scenario 7: Background Tasks

**Objective**: Test long-running tasks with progress updates.

**Steps**:
1. Ask: "Run a background task called 'data migration'"
2. Observe progress updates

**Expected Result**:
- Progress updates appear (working state)
- Final completion message
- Task state transitions are visible

**In Chat**:
```
Tell the agent to run a background task called 'data migration'
```

### Scenario 8: User Interaction & Elicitation

**Objective**: Test input_required state (A2A elicitation flow).

**Steps**:
1. Ask: "Request confirmation for deleting user data"
2. Agent should request confirmation
3. Respond with your choice

**Expected Result**:
- Agent asks for confirmation with danger level
- Input_required state is triggered
- Follow-up is processed correctly

**In Chat**:
```
Ask the agent to request confirmation for deleting all user data
```

### Scenario 9: Error Handling

**Objective**: Test A2A error -> MCP error mapping.

**Steps**:
1. Ask: "Simulate a failure"
2. Observe error handling

**Expected Result**:
- Error is displayed clearly
- Error message is informative
- Client handles error gracefully

**In Chat**:
```
Tell the agent to simulate a failure
```

### Scenario 10: Multi-Step Workflow

**Objective**: Test complex multi-step operations.

**Steps**:
1. Ask: "Process a data pipeline: first sort [5,3,1,4,2], then calculate statistics on the result"
2. Observe multi-step execution

**Expected Result**:
- Agent coordinates multiple sub-agents
- Steps execute in order
- Final result combines outputs

### Scenario 11: State Machine Demo

**Objective**: Test task state transitions.

**Steps**:
1. Ask: "Demonstrate the 'working' state"
2. Ask: "Demonstrate the 'input-required' state"
3. Ask: "Demonstrate the 'completed' state"

**Expected Result**:
- Each state transition is visible
- State messages are appropriate
- Client handles all states

### Scenario 12: Agent Card Inspection

**Objective**: Verify the rich agent card is served correctly.

**Steps**:
1. Curl the agent card directly: `curl http://localhost:9000/.well-known/agent-card.json | jq .`
2. Verify all fields

**Expected Result**:
- 7 skills listed with descriptions and tags
- Capabilities include streaming and stateTransitionHistory
- Extensions include MCP tools, prompts, and resources
- Sub-agent metadata is present

## Verification Checklist

After running all scenarios, verify:

### MCP Protocol
- [ ] Tool discovery works
- [ ] Tool invocation succeeds
- [ ] Parameters are passed correctly
- [ ] Responses are formatted properly
- [ ] Errors are handled gracefully

### A2A Bridge
- [ ] Agent card discovery works
- [ ] Message routing succeeds
- [ ] Streaming responses work
- [ ] Task IDs are tracked correctly

### Agent Capabilities
- [ ] Natural language routing works (no slash commands needed)
- [ ] Calculator sub-agent responds to math queries
- [ ] TextProcessor handles text operations
- [ ] DataProcessor sorts and filters correctly
- [ ] Background tasks show progress
- [ ] Elicitation flows work
- [ ] Error states are handled

### User Experience
- [ ] Responses are timely
- [ ] Streaming feels natural
- [ ] Errors are clear and helpful
- [ ] Agent understands varied phrasing

## Troubleshooting

### No tools showing up

```bash
# Check server logs
docker logs agentique-interactive-mcp-server

# Verify agent is accessible
curl http://localhost:9000/.well-known/agent-card.json

# Restart MCP client
```

### Agent not responding

```bash
# Check if GOOGLE_API_KEY is set
docker exec agentique-interactive-agent env | grep GOOGLE

# Check agent logs
docker logs agentique-interactive-agent

# Verify the agent container is healthy
docker ps | grep agentique-interactive
```

### Timeout errors

```bash
# Check if services are running
docker ps | grep agentique-interactive

# View agent logs for LLM timeouts
docker logs agentique-interactive-agent --tail 30

# Increase timeout in MCP client config
```

### Responses not streaming

- Check MCP client supports streaming
- Verify agent streaming is enabled (check agent card capabilities)
- Review bridge logs for errors

## Advanced Testing

### Testing with Multiple Clients

Test the same scenario across different MCP clients to verify compatibility:

1. Claude CLI
2. Claude Desktop
3. VS Code + Cline
4. GitHub Copilot

Compare:
- Tool discovery UX
- Streaming behavior
- Error formatting
- Multi-turn flow

### Edge Cases

- Very long messages (>10KB)
- Unicode and special characters
- Rapid sequential requests
- Malformed input

## Reporting Issues

If you find issues during interactive testing:

1. Note the MCP client and version
2. Record the exact command/message sent
3. Capture error messages
4. Save relevant logs:
   ```bash
   docker logs agentique-interactive-mcp-server > mcp-server.log
   docker logs agentique-interactive-agent > agent.log
   ```
5. File an issue with all context

## Resources

- MCP client configs: `examples/mcp-clients/`
- Docker compose: `docker-compose.interactive.yml`
- A2A agent code: `a2a_test_agent/src/adk_test_agent/`
- Agent card: `a2a_test_agent/agent_card.json`
- Automated tests: `tests/integration/` and `tests/e2e/`
