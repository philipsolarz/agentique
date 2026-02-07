# Interactive Testing Verification Checklist

Use this checklist when manually testing agentique with real MCP clients.

## Pre-Test Setup

- [ ] Docker is running
- [ ] Ran `./start-interactive.sh` successfully
- [ ] MCP server health check passes (http://localhost:8000/health)
- [ ] Demo agent health check passes (http://localhost:9000/.well-known/agent-card.json)
- [ ] MCP client is configured (Claude CLI / Desktop / VS Code)
- [ ] MCP client recognizes the agentique server

## Basic Connectivity

- [ ] **Tool Discovery**: MCP client lists tools (agents, agent, task, inspect, agent_background)
- [ ] **Agent List**: Can call `agents` tool and see demo agent
- [ ] **Simple Message**: Can send "Hello" to demo agent
- [ ] **Help Command**: `/help` returns full command list

## Core Features

### Request/Response
- [ ] **Echo**: `/echo Test message` returns "Echo: Test message"
- [ ] **Calculator**: `/calc 10 * 5` returns "50"
- [ ] **Context Info**: `/context` shows task_id, context_id, message_id

### Streaming
- [ ] **Stream Command**: `/stream Long text` arrives in chunks
- [ ] **Chunks Arrive**: Can see progressive updates (not all at once)
- [ ] **Complete Response**: All chunks received, response is complete
- [ ] **Multipart**: `/multipart` returns 3 separate parts

### Error Handling
- [ ] **Intentional Error**: `/error` triggers error state
- [ ] **Error Message**: Error is clear and explains it's intentional
- [ ] **Client Recovery**: MCP client handles error gracefully
- [ ] **Can Continue**: Can send more messages after error

### Multi-Turn Conversations
- [ ] **Context Storage**: Send "Remember my name is Alice"
- [ ] **Context Retrieval**: Ask "What did I tell you?" → mentions "Alice"
- [ ] **Memory Command**: `/memory` shows conversation history
- [ ] **Turn Counting**: Agent reports turn numbers correctly
- [ ] **Multiple Turns**: Can have 5+ turn conversation with context

### Long-Running Tasks
- [ ] **Background Task**: `/background process data` shows progress
- [ ] **Progress Updates**: See 0%, 20%, 40%, 60%, 80%, 100%
- [ ] **Completion**: Final "completed successfully" message arrives
- [ ] **Updates Ordered**: Progress appears in correct sequence

### Input Elicitation
- [ ] **Elicitation Start**: `/ask` requests additional input
- [ ] **Clear Instructions**: Agent specifies what input is needed
- [ ] **Follow-up**: Send `CONFIRM: blue 7` as requested
- [ ] **Processing**: Agent processes the confirmation correctly

### Advanced
- [ ] **Slow Response**: `/slow test` shows delayed response handling
- [ ] **No Timeout**: Client waits appropriately (doesn't timeout prematurely)
- [ ] **Context Tracking**: Multiple parallel conversations maintain separate context

## Client-Specific Tests

### Claude CLI
- [ ] `claude mcp list` shows agentique
- [ ] `claude "List available agents"` works
- [ ] Streaming responses appear progressively
- [ ] Multi-turn works across sequential commands

### Claude Desktop
- [ ] Server appears in MCP servers list
- [ ] Can invoke tools via chat
- [ ] Streaming appears natural in UI
- [ ] Errors are formatted nicely

### VS Code (Cline/Continue)
- [ ] Extension recognizes the server
- [ ] Tools are available in autocomplete
- [ ] Can invoke via chat or command palette
- [ ] Streaming works in editor

## Edge Cases

- [ ] **Empty Message**: Sending empty text doesn't crash
- [ ] **Very Long Text**: 1000+ character message handled correctly
- [ ] **Special Characters**: Unicode, emojis, symbols work
- [ ] **Rapid Messages**: Sending 5 messages quickly, all get responses
- [ ] **Unknown Command**: `/unknown` gets helpful error or fallback response

## Performance

- [ ] **Response Time**: Simple echo < 1 second
- [ ] **Streaming Latency**: First chunk < 2 seconds
- [ ] **Background Task**: Completes in ~2-3 seconds
- [ ] **Memory Retrieval**: `/memory` returns instantly

## Integration Points

- [ ] **MCP Protocol**: Client successfully calls tools
- [ ] **A2A Bridge**: Messages route to demo agent
- [ ] **Response Mapping**: A2A responses convert to MCP format
- [ ] **Error Mapping**: A2A errors map to MCP error codes
- [ ] **Streaming Fidelity**: A2A streams map to MCP streams

## User Experience

- [ ] **Help is Clear**: `/help` is easy to understand
- [ ] **Commands Intuitive**: Command syntax makes sense
- [ ] **Errors Helpful**: Error messages suggest next steps
- [ ] **Responses Timely**: No excessive waiting
- [ ] **Overall UX**: Pleasant to interact with

## Documentation Accuracy

- [ ] **INTERACTIVE_TESTING.md**: Instructions work as written
- [ ] **MCP Client Configs**: Example configs work correctly
- [ ] **start-interactive.sh**: Script runs without errors
- [ ] **Demo Scenarios**: All listed scenarios work

## Issues Found

Record any issues discovered during testing:

| Issue | Severity | Steps to Reproduce | Expected | Actual | Notes |
|-------|----------|-------------------|----------|--------|-------|
| Example | High | Send `/calc 5/0` | Error msg | Crash | Division by zero |
|       |          |                   |          |        |       |
|       |          |                   |          |        |       |

## Test Summary

**Date**: ________________
**Tester**: ________________
**MCP Client**: ________________
**Client Version**: ________________
**agentique Version**: ________________

**Tests Passed**: _____ / _____
**Tests Failed**: _____
**Tests Skipped**: _____

**Overall Assessment**:
- [ ] Ready for production
- [ ] Minor issues, can release
- [ ] Major issues, needs work
- [ ] Not ready, significant problems

**Notes**:
_______________________________________________________
_______________________________________________________
_______________________________________________________
_______________________________________________________

## Sign-off

**Tested By**: ________________
**Date**: ________________
**Signature**: ________________
