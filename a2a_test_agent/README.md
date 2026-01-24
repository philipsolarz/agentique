# Enhanced ADK A2A Test Agent

A comprehensive multi-agent system built with Google ADK (Agent Development Kit) and exposed via the A2A (Agent-to-Agent) protocol. This agent serves as a test bed for validating the AgentMCP bridge, demonstrating proper agent routing, context preservation, streaming, and multi-agent orchestration.

## Overview

The test agent implements a sophisticated multi-agent architecture with a root orchestrator and four specialized subagents:

### Agent Architecture

```
TestAgentRoot (Orchestrator)
├── Calculator - Arithmetic operations and statistical analysis
│   ├── add, subtract, multiply, divide
│   └── calculate_statistics
├── DataProcessor - List and data structure manipulation
│   ├── filter_list, sort_list
│   ├── count_occurrences
│   └── process_batch (async)
├── TextProcessor - Text analysis and transformation
│   ├── transform_case, count_words
│   ├── extract_keywords
│   └── reverse_text
└── InfoRetriever - Information lookup and data retrieval
    ├── search_info (mock knowledge base)
    ├── get_random_fact
    └── fetch_data (async)
```

### Key Features

- **Multi-Agent Orchestration**: Root agent intelligently routes requests to appropriate subagents
- **Diverse Tool Types**: Mix of sync/async functions, simple operations, and complex data processing
- **Complex Workflows**: Support for multi-step operations coordinating multiple subagents
- **Async Operations**: Demonstrates proper async/await handling for simulated external calls
- **Rich Data Structures**: Tools that work with lists, dictionaries, and structured data
- **Realistic Testing**: Simulated knowledge bases, data sources, and processing pipelines

## Quick Start

### Run Locally

```bash
# Install dependencies using uv
cd a2a_test_agent
uv pip install -e .

# Configure the server
export A2A_HOST=127.0.0.1
export A2A_PORT=9000
export A2A_PROTOCOL=http
export A2A_BASE_URL=http://127.0.0.1:9000

# REQUIRED: Set your Google API key
export GOOGLE_API_KEY=your_api_key_here

# Optional: Choose a different model
export ADK_MODEL=gemini-2.0-flash

# Start the A2A server
uv run adk-test-agent
```

### Run with Docker Compose (Recommended)

```bash
# From the AgentMCP root directory
export GOOGLE_API_KEY=your_api_key_here

docker compose up --build
```

This starts both the A2A test agent (port 9000) and the MCP server (port 8000).

## API Endpoints

### Agent Card
- **URL**: `http://localhost:9000/.well-known/agent-card.json`
- **Description**: Returns the agent's capabilities, skills, and metadata

### A2A Endpoint
- **URL**: `http://localhost:9000/a2a/TestAgentRoot`
- **Description**: Main A2A communication endpoint for sending messages to the agent

## Example Interactions

### Simple Calculator Operations
```
User: "What is 123 + 456?"
→ Routes to Calculator agent
→ Uses add(123, 456)
→ Returns: 579
```

### Statistical Analysis
```
User: "Calculate statistics for these numbers: 5, 10, 15, 20, 25"
→ Routes to Calculator agent
→ Uses calculate_statistics([5, 10, 15, 20, 25])
→ Returns: {"count": 5, "mean": 15, "median": 15, "min": 5, "max": 25, "sum": 75}
```

### Data Processing
```
User: "Sort these words alphabetically: zebra, apple, mango, banana"
→ Routes to DataProcessor agent
→ Uses sort_list(["zebra", "apple", "mango", "banana"])
→ Returns: ["apple", "banana", "mango", "zebra"]
```

### Text Manipulation
```
User: "Convert 'hello world' to title case"
→ Routes to TextProcessor agent
→ Uses transform_case("hello world", "title")
→ Returns: "Hello World"
```

### Information Retrieval
```
User: "Tell me about Python"
→ Routes to InfoRetriever agent
→ Uses search_info("Python")
→ Returns: Information about Python programming language
```

### Complex Multi-Step Workflows
```
User: "Calculate the sum of 10, 20, 30 and then tell me if it's in the knowledge base"
→ Routes to Calculator agent → add numbers
→ Routes to InfoRetriever agent → search for result
→ Returns: Combined results from both agents
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `A2A_HOST` | `0.0.0.0` | Server bind address |
| `A2A_PORT` | `9000` | Server port |
| `A2A_PROTOCOL` | `http` | Protocol (http/https) |
| `A2A_BASE_URL` | - | Full base URL (important for Docker) |
| `A2A_RELOAD` | `false` | Enable uvicorn auto-reload |
| `A2A_AGENT_CARD` | - | Path to custom agent card JSON |
| `GOOGLE_API_KEY` | - | **Required** Google AI API key |
| `ADK_MODEL` | `gemini-2.0-flash` | Gemini model to use |

## Testing the Agent

### Test Agent Card
```bash
curl http://localhost:9000/.well-known/agent-card.json | jq
```

### Test A2A Communication
```bash
curl -X POST http://localhost:9000/a2a/TestAgentRoot \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [{"text": "What is 5 + 3?"}]
    }]
  }'
```

## Architecture Details

### Agent Design Principles

1. **Separation of Concerns**: Each subagent handles a specific domain
2. **Clear Instructions**: Agents have well-defined instructions for routing and execution
3. **Tool Diversity**: Mix of simple functions, complex operations, and async calls
4. **Realistic Behavior**: Simulated delays, external calls, and data processing

### A2A Integration

The agent uses Google ADK's `to_a2a()` utility to expose the agent hierarchy via the A2A protocol:

- **Agent Card Generation**: Automatically generates card from agent metadata
- **Request Handling**: Routes A2A messages to the appropriate agent
- **Context Preservation**: Maintains conversation context across agent transfers
- **Streaming Support**: Enables incremental response streaming when supported

### Tool Implementation Patterns

#### Synchronous Tools
```python
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b
```

#### Asynchronous Tools
```python
async def fetch_data(source: str) -> dict[str, Any]:
    """Simulate fetching data from an external source."""
    await asyncio.sleep(0.2)  # Simulate network delay
    return {"source": source, "data": {...}}
```

#### Complex Data Processing
```python
def calculate_statistics(numbers: list[float]) -> dict[str, float]:
    """Calculate statistics for a list of numbers."""
    return {
        "mean": sum(numbers) / len(numbers),
        "median": ...,
        ...
    }
```

## Development

### Project Structure
```
a2a_test_agent/
├── src/
│   └── adk_test_agent/
│       ├── __init__.py
│       ├── agent.py         # Agent definitions and tools
│       └── server.py        # A2A server setup
├── pyproject.toml           # Dependencies and build config
└── README.md               # This file
```

### Adding New Subagents

1. Define tools as functions in `agent.py`
2. Create an Agent instance with name, description, instructions, and tools
3. Add the agent to the root agent's `sub_agents` list
4. Update the root agent's instruction to mention the new capability

### Modifying Tools

Tools should:
- Have clear docstrings (used by the LLM to understand the tool)
- Use type hints for all parameters
- Return structured data when appropriate
- Handle errors gracefully

## Troubleshooting

### Agent Not Starting
- Ensure `GOOGLE_API_KEY` is set and valid
- Check that port 9000 is not already in use
- Verify all dependencies are installed with `uv pip install -e .`

### Agent Card Not Accessible
- Check that the server is running: `curl http://localhost:9000/.well-known/agent-card.json`
- Verify A2A_BASE_URL is set correctly for Docker environments

### Agent Not Responding
- Check server logs for errors
- Verify the model name is correct (see available models at https://ai.google.dev/models)
- Ensure API key has proper permissions

### Docker Issues
- Make sure GOOGLE_API_KEY is exported before running `docker compose up`
- Check container logs: `docker compose logs a2a-test-agent`
- Verify network connectivity between containers

## References

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [A2A Protocol Specification](https://github.com/google-a2a/A2A/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Agent Development Best Practices](https://google.github.io/adk-docs/agents/overview/)
