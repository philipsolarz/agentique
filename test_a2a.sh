#!/bin/bash
# Test A2A agent with JSON-RPC format

echo "🧪 AgentMCP A2A Test Suite"
echo "=========================================="
echo ""

# Test 1: Agent Card
echo "✓ Testing Agent Card..."
CARD_NAME=$(curl -s http://localhost:9000/.well-known/agent-card.json | jq -r '.name')
if [ "$CARD_NAME" = "TestAgentRoot" ]; then
    echo "  ✅ Agent card OK: $CARD_NAME"
else
    echo "  ❌ Agent card FAILED: $CARD_NAME"
    exit 1
fi

# Test 2: Simple Calculation
echo "✓ Testing Simple Calculation (25 + 17)..."
CALC_RESULT=$(curl -s -X POST http://localhost:9000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "What is 25 + 17?"}],
        "message_id": "test-calc"
      }
    },
    "id": 1
  }' | jq -r '.result.status.state')

if [ "$CALC_RESULT" = "completed" ]; then
    echo "  ✅ Calculation completed successfully"
else
    echo "  ❌ Calculation failed: $CALC_RESULT"
    exit 1
fi

# Test 3: Multi-Agent Routing (should route to Calculator)
echo "✓ Testing Multi-Agent Routing..."
AGENT_NAME=$(curl -s -X POST http://localhost:9000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Calculate statistics for: 5, 10, 15"}],
        "message_id": "test-routing"
      }
    },
    "id": 2
  }' | jq -r '.result.metadata.adk_author')

if [ "$AGENT_NAME" = "Calculator" ]; then
    echo "  ✅ Routed to correct agent: $AGENT_NAME"
else
    echo "  ❌ Wrong agent: $AGENT_NAME (expected Calculator)"
fi

# Test 4: Text Processing (should route to TextProcessor)
echo "✓ Testing Text Processing..."
TEXT_RESULT=$(curl -s -X POST http://localhost:9000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Reverse the word: hello"}],
        "message_id": "test-text"
      }
    },
    "id": 3
  }' | jq -r '.result.artifacts[0].parts[0].text')

echo "  ✅ Text processing result: $TEXT_RESULT"

echo ""
echo "=========================================="
echo "🎉 A2A Test Suite Complete!"
echo "=========================================="
