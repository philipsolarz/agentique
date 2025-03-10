"""
Example usage of the refactored Agentique library.

This example demonstrates how to use pydantic models for tool registration
and create an agent that can use these tools.
"""

import os
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
from dotenv import load_dotenv

from agentique import Agentique, StructuredResult

load_dotenv()


# Define some Pydantic models for tool parameters
class WeatherParams(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")
    units: Optional[str] = Field(..., description="Temperature units (celsius or fahrenheit)")


class SearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    num_results: int = Field(..., description="Number of results to return")


class DistanceParams(BaseModel):
    origin: str = Field(..., description="Starting location")
    destination: str = Field(..., description="Ending location")


# Define a Pydantic model for structured output
class TravelPlan(StructuredResult):
    destination: str = Field(..., description="Travel destination")
    activities: List[str] = Field(..., description="List of activities")
    weather_forecast: str = Field(..., description="Weather forecast")
    estimated_cost: float = Field(..., description="Estimated cost in USD")


# Initialize the Agentique library
openai_api_key = os.environ.get("OPENAI_API_KEY")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

agentique = Agentique(
    openai_api_key=openai_api_key,
    anthropic_api_key=anthropic_api_key
)


# Define tool functions and register them with Pydantic models
def get_weather(latitude: float, longitude: float, units: str = "celsius") -> str:
    """Get the current weather for a location."""
    # In a real implementation, this would call a weather API
    temp = 25 if units == "celsius" else 77
    return f"The current temperature at coordinates ({latitude}, {longitude}) is {temp}°{units[0].upper()}."

agentique.register_tool(
    name="get_weather",
    function=get_weather,
    parameter_model=WeatherParams
)


def search_web(query: str, num_results: int = 5) -> str:
    """Search the web for information."""
    # In a real implementation, this would call a search API
    return f"Found {num_results} results for '{query}':\n" + \
           "\n".join([f"Result {i+1}: Sample result about {query}" for i in range(num_results)])

agentique.register_tool(
    name="search_web",
    function=search_web,
    parameter_model=SearchParams
)


def calculate_distance(origin: str, destination: str) -> str:
    """Calculate the distance between two locations."""
    # In a real implementation, this would call a mapping API
    return f"The distance from {origin} to {destination} is 500 km."

agentique.register_tool(
    name="calculate_distance",
    function=calculate_distance,
    parameter_model=DistanceParams
)


# Create a travel planning agent that can use the tools
travel_agent = agentique.create_agent(
    agent_id="travel_planner",
    system_prompt="""You are a travel planning assistant. 
    Help users plan their trips by providing information about destinations, 
    weather forecasts, and activities. Use the available tools to gather information.""",
    model="gpt-4o-mini",  # Use a smaller/cheaper model for testing
    structured_output_model=TravelPlan  # Use the structured output model
)


async def run_example():
    """Run the example agent with a sample query."""
    # Run the agent with a sample query
    response = await travel_agent.run(
        """I'm planning a trip to Barcelona next week. 
        Can you suggest some activities and tell me what the weather might be like?
        Also, how far is it from Madrid to Barcelona?""",
        # Specify which tools the agent can use for this query
        tools=["get_weather", "search_web", "calculate_distance"]
    )
    
    # Print the response
    if isinstance(response, TravelPlan):
        print("\nStructured Response:")
        print(f"Destination: {response.destination}")
        print(f"Activities: {', '.join(response.activities)}")
        print(f"Weather: {response.weather_forecast}")
        print(f"Estimated Cost: ${response.estimated_cost}")
    else:
        print("\nText Response:")
        print(response)


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_example())