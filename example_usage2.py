"""
Example usage of the Agentique library with improved function calling.

This example demonstrates how to create complex tools with advanced parameter models,
register them with the ToolRegistry, and use them in an Agent.
"""

import asyncio
import os
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Any, Literal
from datetime import datetime
import json
from dotenv import load_dotenv

from rich import print
from rich.panel import Panel

from agentique import (
    Agent, 
    ToolRegistry, 
    OpenAIClient, 
    AgentConfig
)

load_dotenv()

# -------------------------------
# Advanced Parameter Models
# -------------------------------

class StatusEnum(str, Enum):
    """Status enum for items in the analysis."""
    active = "active"
    inactive = "inactive"
    pending = "pending"

class LevelEnum(str, Enum):
    """Level enum for priority indication."""
    low = "low"
    medium = "medium"
    high = "high"

class Detail(BaseModel):
    """Detailed information about an item."""
    description: str = Field(..., description="Detailed description of the item.")
    # parameters: Dict[str, Union[str, int, float]] = Field(
    #     ..., description="Arbitrary parameters related to the item."
    # )
    rating: float = Field(..., ge=0, le=5, description="Rating between 0 and 5.")

class Item(BaseModel):
    """An item to be analyzed."""
    name: str = Field(..., description="Name of the item.")
    value: Union[int, float] = Field(..., description="Numerical value for the item.")
    flags: List[bool] = Field(..., description="List of boolean flags.")
    detail: Detail = Field(..., description="Detailed information for the item.")
    quantity: int = Field(..., gt=0, description="Quantity must be greater than 0.")

class TreeNode(BaseModel):
    """A node in a recursive tree structure."""
    node_id: str = Field(..., description="Unique identifier for the node.")
    children: Optional[List["TreeNode"]] = Field(None, description="Optional list of child nodes.")
TreeNode.model_rebuild()  # Required for recursive Pydantic models

class ConfigObject(BaseModel):
    """Configuration object with detailed settings."""
    mode: str = Field(..., description="Mode for configuration.")
    options: Dict[str, Any] = Field(..., description="Additional configuration options.")

class AdvancedAnalysisParams(BaseModel):
    """
    Parameters for advanced analysis with complex nested structures.
    
    This demonstrates the full capabilities of the schema processor,
    handling nested objects, recursive structures, enums, and unions.
    """
    id: str = Field(..., description="Unique identifier for the analysis.")
    # title: str = Field(..., description="Title of the analysis.")
    status: StatusEnum = Field(..., description="Current status of the analysis.")
    metadata: Optional[Dict[str, Union[str, int, bool]]] = Field(None, description="Optional metadata.")
    tags: Optional[List[str]] = Field(None, description="List of tags for the analysis.")
    items: List[Item] = Field(..., description="List of items to analyze.")
    tree: Optional[List[TreeNode]] = Field(None, description="A recursive tree of nodes.")
    config: Union[str, ConfigObject] = Field(..., description="Configuration as a simple string or a detailed object.")
    levels: List[LevelEnum] = Field(..., description="List of level indicators.")
    priority: int = Field(..., ge=1, le=5, description="Priority between 1 and 5.")
    score: float = Field(..., ge=0, le=100, description="Score between 0 and 100.")
    discount: float = Field(0.0, ge=0, le=1, description="Discount rate as a fraction between 0 and 1.")

class WeatherParams(BaseModel):
    """Parameters for the weather forecast tool."""
    location: str = Field(..., description="City or location to get weather for.")
    days: int = Field(3, ge=1, le=7, description="Number of days for the forecast (1-7).")
    units: Literal["metric", "imperial"] = Field("metric", description="Units for temperature (metric or imperial).")

class SearchParams(BaseModel):
    """Parameters for the search tool."""
    query: str = Field(..., description="Search query string.")
    limit: int = Field(5, ge=1, le=20, description="Maximum number of results to return.")

# -------------------------------
# Tool Functions
# -------------------------------

async def advanced_analysis(
    id: str,
    # title: str,
    status: StatusEnum,
    items: List[Item],
    config: Union[str, ConfigObject],
    levels: List[LevelEnum],
    priority: int,
    score: float,
    discount: float = 0.0,
    metadata: Optional[Dict[str, Union[str, int, bool]]] = None,
    tags: Optional[List[str]] = None,
    tree: Optional[List[TreeNode]] = None
) -> Dict[str, Any]:
    """
    Performs an advanced analysis based on the provided complex parameters.
    
    This function demonstrates the full capabilities of our function calling schema,
    including nested objects, recursive structures, unions, and enums.
    """
    # In a real scenario, this would perform actual analysis
    # Here we'll just return a summary of the input
    
    result = {
        "analysis_id": id,
        # "title": title,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "summary": f"Analysis of {len(items)} items with priority {priority}",
        "item_count": len(items),
        "average_value": sum(item["value"] for item in items) / len(items),
        "average_rating": sum(item["detail"]["rating"] for item in items) / len(items),
        "total_quantity": sum(item["quantity"] for item in items),
        "priority_level": priority,
        "config_type": "object" if isinstance(config, ConfigObject) else "string"
    }
    
    # Simulate some processing time
    await asyncio.sleep(0.5)
    
    return result

async def get_weather(
    location: str,
    days: int = 3,
    units: Literal["metric", "imperial"] = "metric"
) -> Dict[str, Any]:
    """
    Gets weather forecast for the specified location.
    
    Args:
        location: City or location to get weather for
        days: Number of days for the forecast (1-7)
        units: Units for temperature (metric or imperial)
    
    Returns:
        Weather forecast data
    """
    # Simulate API call to weather service
    cities = {
        "new york": {"temp": 22, "condition": "Partly Cloudy", "humidity": 65},
        "london": {"temp": 18, "condition": "Rainy", "humidity": 80},
        "tokyo": {"temp": 26, "condition": "Sunny", "humidity": 70},
        "sydney": {"temp": 30, "condition": "Clear", "humidity": 55},
        "paris": {"temp": 20, "condition": "Cloudy", "humidity": 60}
    }
    
    location_lower = location.lower()
    
    # Generate simulated forecast
    if location_lower in cities:
        base = cities[location_lower]
    else:
        base = {"temp": 25, "condition": "Unknown", "humidity": 60}
    
    # Apply unit conversion if needed
    if units == "imperial":
        base["temp"] = round((base["temp"] * 9/5) + 32, 1)
        temp_unit = "°F"
    else:
        temp_unit = "°C"
    
    # Generate daily forecasts with slight variations
    forecast = []
    import random
    for i in range(days):
        temp_variation = random.uniform(-3, 3)
        day_forecast = {
            "day": i + 1,
            "date": (datetime.now().date().replace(day=datetime.now().day + i)).isoformat(),
            "temp": round(base["temp"] + temp_variation, 1),
            "temp_unit": temp_unit,
            "condition": base["condition"],
            "humidity": min(100, max(0, base["humidity"] + random.randint(-10, 10)))
        }
        forecast.append(day_forecast)
    
    # Simulate API delay
    await asyncio.sleep(0.3)
    
    return {
        "location": location,
        "units": units,
        "current": {
            "temp": base["temp"],
            "temp_unit": temp_unit,
            "condition": base["condition"],
            "humidity": base["humidity"]
        },
        "forecast": forecast
    }

async def search_info(
    query: str,
    limit: int = 5
) -> List[Dict[str, str]]:
    """
    Search for information based on the query.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
    
    Returns:
        List of search results
    """
    # Simulate search results
    topics = {
        "python": [
            {"title": "Python Official Website", "url": "https://www.python.org", "snippet": "The official home of the Python Programming Language."},
            {"title": "Python on Wikipedia", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)", "snippet": "Python is a high-level, general-purpose programming language."},
            {"title": "Learn Python - Codecademy", "url": "https://www.codecademy.com/learn/learn-python", "snippet": "Learn Python, a powerful language used by sites like YouTube and Dropbox."},
            {"title": "Python for Beginners", "url": "https://www.pythonforbeginners.com/", "snippet": "Python tutorials for beginners to help you learn Python programming."},
            {"title": "Real Python", "url": "https://realpython.com/", "snippet": "Learn Python online: Python tutorials for developers of all skill levels."}
        ],
        "ai": [
            {"title": "Artificial Intelligence - Wikipedia", "url": "https://en.wikipedia.org/wiki/Artificial_intelligence", "snippet": "Artificial intelligence (AI) is intelligence demonstrated by machines."},
            {"title": "What is AI? | IBM", "url": "https://www.ibm.com/topics/artificial-intelligence", "snippet": "Artificial intelligence is a field of science concerned with building computers and machines that can reason, learn, and act."},
            {"title": "Stanford AI Lab", "url": "https://ai.stanford.edu/", "snippet": "Stanford Artificial Intelligence Laboratory is a center of excellence for AI research."},
            {"title": "MIT AI Lab", "url": "https://www.csail.mit.edu/", "snippet": "The MIT Computer Science and Artificial Intelligence Laboratory pioneers research in computing."},
            {"title": "Google AI", "url": "https://ai.google/", "snippet": "Building helpful AI tools and technologies that benefit people everyday."}
        ],
        "programming": [
            {"title": "Stack Overflow", "url": "https://stackoverflow.com/", "snippet": "Stack Overflow is the largest, most trusted online community for developers."},
            {"title": "GitHub", "url": "https://github.com/", "snippet": "GitHub is where over 83 million developers shape the future of software."},
            {"title": "Codecademy", "url": "https://www.codecademy.com/", "snippet": "Learn to code for free with hands-on projects and learning paths."},
            {"title": "freeCodeCamp", "url": "https://www.freecodecamp.org/", "snippet": "Learn to code for free with thousands of interactive tutorials."},
            {"title": "MDN Web Docs", "url": "https://developer.mozilla.org/", "snippet": "Resources for developers, by developers."}
        ]
    }
    
    # Find the most relevant topic or default to programming
    query_lower = query.lower()
    results = []
    
    for topic, topic_results in topics.items():
        if topic in query_lower:
            results = topic_results
            break
    
    if not results:
        results = topics["programming"]  # Default to programming
    
    # Limit results
    limited_results = results[:limit]
    
    # Simulate search delay
    await asyncio.sleep(0.3)
    
    return limited_results

# -------------------------------
# Run Example
# -------------------------------

async def run_example():
    # Initialize OpenAI client (using API key from environment)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[bold red]Error: OPENAI_API_KEY environment variable is required.[/bold red]")
        return

    client = OpenAIClient(api_key=api_key, model="gpt-4o-mini")
    
    # Initialize tool registry
    registry = ToolRegistry()
    
    # Register tools with improved schema generation
    registry.register(advanced_analysis, AdvancedAnalysisParams)
    registry.register(get_weather, WeatherParams)
    registry.register(search_info, SearchParams)
    
    # Initialize agent
    agent_config = AgentConfig(
        name="AgentiqueDemo",
        model="gpt-4o-mini",
        system_prompt="""You are a helpful assistant with access to various tools.
You can use these tools to provide information and perform analyses.
Always provide thoughtful and detailed responses based on the results from tools."""
    )
    
    agent = Agent(config=agent_config, client=client, tool_registry=registry)
    
    # Example interactions
    examples = [
        "What's the weather like in London for the next 5 days?",
        "Can you search for information about Python programming?",
        "Perform an advanced analysis with the following parameters: id='demo1', title='Demo Analysis', status='active', items=[{name:'Item1', value:42, flags:[true, false], detail:{description:'Test item', parameters:{'key1':'value1'}, rating:4.5}, quantity:2}], config='simple', levels=['low', 'medium'], priority=2, score=75"
    ]
    
    # Run examples
    for i, example in enumerate(examples):
        print(Panel.fit(f"[bold cyan]Example {i+1}:[/bold cyan] {example}", border_style="cyan"))
        
        # Run the agent
        response = await agent.run(
            user_input=example,
            tools=["advanced_analysis", "get_weather", "search_info"],
            max_iterations=3
        )
        
        # Print the response
        print(Panel.fit(f"[bold green]Response:[/bold green]\n{response}", border_style="green"))
        print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    # Run the async example
    asyncio.run(run_example())