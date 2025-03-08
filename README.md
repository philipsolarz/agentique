# Agentique

Agentique is a Python framework for creating agentic AI systems that can interact with their environment through function calls and communicate with other agents.

## Features

- **Multi-model support**: Use OpenAI or Anthropic models
- **Tool integration**: Enable AI agents to call functions, APIs, and services
- **Agent communication**: Allow agents to message and collaborate with each other
- **Structured outputs**: Define custom structured output formats using Pydantic models
- **Conversation management**: Maintain and manipulate conversation history
- **Robust error handling**: Retry logic for API calls with exponential backoff

## Installation

```bash
pip install agentique
```

## Quick Start

```python
import asyncio
import os
from pydantic import BaseModel, Field
from agentique import Agentique, StructuredResult

# Define a custom output structure (optional)
class AnalysisResult(StructuredResult):
    summary: str = Field(..., description="Brief summary of the analysis")
    key_points: list[str] = Field(..., description="List of key points")
    sentiment: str = Field(..., description="Overall sentiment")

# Create a custom tool
async def get_data(query: str):
    """Get data based on a query"""
    # Implementation would call an API or database
    return {"data": f"Results for: {query}"}

async def main():
    # Initialize Agentique with API keys
    agentique = Agentique(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")  # Optional
    )
    
    # Register custom tools
    agentique.register_tool(
        name="get_data",
        function=get_data,
        description="Get data based on a query string"
    )
    
    # Create an agent with a custom output structure
    agent = agentique.create_agent(
        agent_id="analyst",
        system_prompt="You are a data analyst. Analyze data and provide insights.",
        structured_output_model=AnalysisResult
    )
    
    # Run the agent with a query
    result = await agent.run(
        user_input="Analyze the latest market trends",
        tools=["get_data"]
    )
    
    # Handle structured output
    if isinstance(result, AnalysisResult):
        print(f"Summary: {result.summary}")
        print(f"Key Points: {result.key_points}")
        print(f"Sentiment: {result.sentiment}")
    else:
        print(f"Response: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Components

- **Agentique**: Main interface for creating and managing agents
- **Agent**: Handles interactions with AI models, maintains conversation history
- **ToolRegistry**: Manages available tools that agents can use
- **StructuredResult**: Base model for structured outputs
- **OpenAIClientWrapper/AnthropicClientWrapper**: API client wrappers with retry logic

## Customization

### Custom Output Structures

Define your own structured output format by subclassing `StructuredResult`:

```python
from pydantic import BaseModel, Field
from agentique import StructuredResult

class ResearchReport(StructuredResult):
    title: str = Field(..., description="Report title")
    findings: list[str] = Field(..., description="Key research findings")
    recommendations: list[str] = Field(..., description="Recommended actions")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level")
```

### Custom Tools

Register custom functions as tools for your agents:

```python
# Synchronous tool
def calculate_metric(data: list[float], method: str = "mean"):
    """Calculate statistical metrics from a list of values"""
    if method == "mean":
        return sum(data) / len(data)
    elif method == "max":
        return max(data)
    # ...etc.

# Register the tool
agentique.register_tool(
    name="calculate_metric",
    function=calculate_metric,
    description="Calculate statistical metrics from data"
)
```

## Multi-Agent Systems

Create multiple agents that can communicate with each other:

```python
# Create agents with different roles
researcher = agentique.create_agent(
    agent_id="researcher",
    system_prompt="You research facts and information."
)

writer = agentique.create_agent(
    agent_id="writer",
    system_prompt="You write engaging content based on research."
)

# Have agents communicate
from agentique import message_agent

async def coordinate_agents():
    # Get research from the researcher
    research_result = await message_agent(
        target_agent_id="researcher",
        message="Research the history of artificial intelligence."
    )
    
    # Pass the research to the writer
    content = await message_agent(
        target_agent_id="writer",
        message=f"Write an article based on this research: {research_result}"
    )
    
    return content
```

## API Reference

For full API documentation, see the [API Reference](https://agentique.readthedocs.io/).

## License

MIT