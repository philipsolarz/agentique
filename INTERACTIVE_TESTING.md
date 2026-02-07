# Interactive Testing Guide

This guide walks you through manual testing of agentique using real MCP clients (Claude CLI, GitHub Copilot, VS Code extensions, etc.).

## Why Interactive Testing?

Automated tests validate protocol conformance, but interactive testing with real MCP clients ensures:
- ✅ Real-world MCP client compatibility
- ✅ User experience quality
- ✅ Streaming behavior in production
- ✅ Multi-turn conversation flows
- ✅ Error handling from a user perspective
- ✅ Tool discovery and invocation UX

## Quick Start

### 1. Start the Interactive Stack

```bash
# From the agentique repository root
docker compose -f docker-compose.interactive.yml up -d --wait
```

Verify services are healthy:
```bash
# Check MCP server
curl http://localhost:8000/health

# Check demo agent
curl http://localhost:9000/.well-known/agent-card.json
```

### 2. Configure Your MCP Client

Choose your preferred MCP client and follow the configuration guide in `examples/mcp-clients/README.md`:

- **Claude CLI**: `~/.claude/mcp_settings.json`
- **Claude Desktop**: Platform-specific config file
- **VS Code (Cline/Continue)**: `.vscode/settings.json`

### 3. Run Test Scenarios

Follow the scenarios below to systematically test all features.

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
2. Verify the demo agent appears

**Expected Result**:
- Response includes "demo-agent"
- Agent capabilities are shown
- Skills are listed

**In Chat**:
```
List all available agents
```

or

```
Call the agents tool
```

### Scenario 3: Simple Message to Agent

**Objective**: Test basic message routing from MCP → Bridge → A2A.

**Steps**:
1. Send: "Hello, demo agent!"
2. Verify you get a response

**Expected Result**:
- Response acknowledges the message
- Mentions conversation turn number
- Suggests using `/help`

**In Chat**:
```
Send a message to the demo agent: Hello, demo agent!
```

or use the `/help` command:

```
Send this to the demo agent: /help
```

### Scenario 4: Command Help

**Objective**: Verify the demo agent's help system.

**Steps**:
1. Send: `/help` to the demo agent
2. Review the command list

**Expected Result**:
- All commands are listed
- Examples are provided
- Formatting is readable

**In Chat**:
```
Tell the demo agent: /help
```

### Scenario 5: Echo Test

**Objective**: Test basic request/response flow.

**Steps**:
1. Send: `/echo This is a test message`
2. Verify echoed response

**Expected Result**:
- Response is: "Echo: This is a test message"

**In Chat**:
```
Send to demo agent: /echo This is a test message
```

### Scenario 6: Streaming Response

**Objective**: Test streaming responses (critical for A2A).

**Steps**:
1. Send: `/stream Tell me a long story`
2. Observe if response arrives in chunks

**Expected Result**:
- Response streams incrementally (you should see chunks appear)
- All chunks arrive
- Final response is complete

**In Chat**:
```
Send to demo agent: /stream Tell me a long story about testing
```

### Scenario 7: Error Handling

**Objective**: Test A2A error → MCP error mapping.

**Steps**:
1. Send: `/error` to the demo agent
2. Observe error handling

**Expected Result**:
- Error is displayed clearly
- Error message explains it's intentional
- Client handles error gracefully

**In Chat**:
```
Tell demo agent: /error
```

### Scenario 8: Multi-Turn Conversation

**Objective**: Test conversation context maintenance.

**Steps**:
1. Send: "Remember that my name is Alice"
2. Then send: "What did I just tell you?"
3. Verify agent remembers

**Expected Result**:
- Agent confirms it remembers "Alice"
- Shows conversation history
- Context is maintained across turns

**In Chat**:
```
Tell demo agent: Remember that my name is Alice
```

Wait for response, then:

```
Ask demo agent: What did I just tell you?
```

### Scenario 9: Conversation Memory

**Objective**: Test the `/memory` command.

**Steps**:
1. Have a few exchanges with the agent
2. Send: `/memory`
3. Review conversation history

**Expected Result**:
- Shows last N messages
- Includes timestamps
- Both user and assistant messages visible

**In Chat**:
```
Tell demo agent: /memory
```

### Scenario 10: Calculator

**Objective**: Test simple computation.

**Steps**:
1. Send: `/calc 15 * 7`
2. Verify calculation

**Expected Result**:
- Result: "15 * 7 = 105"

**In Chat**:
```
Tell demo agent: /calc 15 * 7
```

### Scenario 11: Background Task

**Objective**: Test long-running tasks with progress updates.

**Steps**:
1. Send: `/background data processing`
2. Observe progress updates

**Expected Result**:
- Progress updates appear (0%, 20%, 40%, ...)
- Final completion message
- All updates arrive in order

**In Chat**:
```
Tell demo agent: /background process large dataset
```

### Scenario 12: Input Elicitation

**Objective**: Test input_required state (A2A elicitation flow).

**Steps**:
1. Send: `/ask`
2. Agent requests additional input
3. Send: `CONFIRM: blue 7`
4. Verify agent processes the input

**Expected Result**:
- Agent asks for color and number
- Waits for confirmation
- Processes the follow-up correctly

**In Chat**:
```
Tell demo agent: /ask
```

Then after it asks:

```
Tell demo agent: CONFIRM: blue 7
```

### Scenario 13: Context Information

**Objective**: Test context ID tracking.

**Steps**:
1. Send: `/context`
2. Review context information

**Expected Result**:
- Shows task_id
- Shows context_id
- Shows message_id

**In Chat**:
```
Tell demo agent: /context
```

### Scenario 14: Slow Response

**Objective**: Test timeout handling.

**Steps**:
1. Send: `/slow test timeout`
2. Observe delayed response

**Expected Result**:
- Initial acknowledgment appears
- After ~2 seconds, full response arrives
- Client handles the delay gracefully

**In Chat**:
```
Tell demo agent: /slow testing timeouts
```

### Scenario 15: Multipart Response

**Objective**: Test multiple artifacts in one response.

**Steps**:
1. Send: `/multipart`
2. Observe multiple parts

**Expected Result**:
- Three separate parts appear
- Each part arrives with small delay
- All parts are received

**In Chat**:
```
Tell demo agent: /multipart
```

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
- [ ] Multi-turn context is maintained
- [ ] Task IDs are tracked correctly

### User Experience
- [ ] Responses are timely
- [ ] Streaming feels natural
- [ ] Errors are clear and helpful
- [ ] Commands are intuitive
- [ ] Help system is useful

### Advanced Features
- [ ] Background tasks show progress
- [ ] Input elicitation flows work
- [ ] Conversation memory persists
- [ ] Context tracking is accurate
- [ ] Timeouts are handled

## Troubleshooting

### No tools showing up

```bash
# Check server logs
docker logs agentique-interactive-mcp-server

# Verify agent is accessible
curl http://localhost:9000/.well-known/agent-card.json

# Restart MCP client
```

### Timeout errors

```bash
# Check if services are running
docker ps | grep agentique-interactive

# View demo agent logs
docker logs agentique-interactive-demo-agent

# Increase timeout in MCP client config
```

### Responses not streaming

- Check MCP client supports streaming
- Verify demo agent streaming is enabled
- Review bridge logs for errors

### Context not maintained

- Verify context_id is being passed
- Check demo agent memory logs
- Ensure same session is used

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

### Load Testing

Send rapid-fire messages to test:
- Concurrent request handling
- Context isolation
- Memory management

### Edge Cases

- Very long messages (>10KB)
- Unicode and special characters
- Rapid context switching
- Malformed commands

## Reporting Issues

If you find issues during interactive testing:

1. Note the MCP client and version
2. Record the exact command/message sent
3. Capture error messages
4. Save relevant logs:
   ```bash
   docker logs agentique-interactive-mcp-server > mcp-server.log
   docker logs agentique-interactive-demo-agent > demo-agent.log
   ```
5. File an issue with all context

## Next Steps

After validating basic functionality:
1. Test with your own custom A2A agents
2. Configure routing strategies
3. Add middleware for observability
4. Test push notifications (if supported)
5. Validate against production agents

## Resources

- MCP client configs: `examples/mcp-clients/`
- Docker compose: `docker-compose.interactive.yml`
- Demo agent code: `tests/agents/demo_agent.py`
- Automated tests: `tests/integration/` and `tests/e2e/`
