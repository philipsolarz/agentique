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
