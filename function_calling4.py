import openai
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

# Import the built-in helper from OpenAI's library.
from openai.lib._tools import pydantic_function_tool

load_dotenv()

# Set up debug logging.
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global registry for functions registered as tools.
# -----------------------------------------------------------------------------
FUNCTION_REGISTRY: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Decorator to register an agent function using the built-in tool.
# -----------------------------------------------------------------------------
def agent_function(model: Type[BaseModel]):
    def decorator(func: Callable):
        func_description = (func.__doc__ or "No description provided.").strip()
        # Generate the tool schema using OpenAI's built-in helper.
        tool_schema = pydantic_function_tool(model, name=func.__name__, description=func_description)
        FUNCTION_REGISTRY[func.__name__] = {
            "function": func,
            "schema": tool_schema,
            "model": model,
        }
        logger.debug("Registered function '%s' with schema: %s", func.__name__, tool_schema)
        return func
    return decorator

# -----------------------------------------------------------------------------
# Basic example Pydantic models for function parameters.
# -----------------------------------------------------------------------------
class WeatherParams(BaseModel):
    latitude: float = Field(..., description="Latitude of the location.")
    longitude: float = Field(..., description="Longitude of the location.")

class EmailParams(BaseModel):
    to: str = Field(..., description="Recipient email address.")
    body: str = Field(..., description="Content of the email.")

class SortOptions(str, Enum):
    relevance = "relevance"
    date = "date"
    popularity = "popularity"
    alphabetical = "alphabetical"

class KBOptions(BaseModel):
    num_results: int = Field(..., description="Number of top results to return.")
    domain_filter: str = Field(..., description="Optional domain to narrow the search (e.g. 'finance', 'medical').")
    sort_by: SortOptions = Field(..., description="How to sort results.")

class KBParams(BaseModel):
    query: str = Field(..., description="The user question or search query.")
    options: KBOptions

# -----------------------------------------------------------------------------
# Advanced example Pydantic models for flight booking.
# -----------------------------------------------------------------------------
class Passenger(BaseModel):
    first_name: str = Field(..., description="Passenger's first name.")
    last_name: str = Field(..., description="Passenger's last name.")
    age: Optional[int] = Field(None, description="Passenger's age.")
    passport_number: Optional[str] = Field(None, description="Passenger's passport number.")

class FlightDetails(BaseModel):
    departure: str = Field(..., description="Departure airport code.")
    arrival: str = Field(..., description="Arrival airport code.")
    date: str = Field(..., description="Flight date in YYYY-MM-DD format.")

class SeatPreference(str, Enum):
    aisle = "aisle"
    window = "window"
    middle = "middle"

class AdvancedBookingParams(BaseModel):
    flight: FlightDetails
    passengers: List[Passenger]
    seat_preference: Optional[SeatPreference] = Field(None, description="Preferred seat type.")
    meal_preference: Optional[str] = Field(None, description="Meal preference during flight.")

# -----------------------------------------------------------------------------
# Example functions registered as agent functions.
# -----------------------------------------------------------------------------
@agent_function(model=WeatherParams)
def get_weather(latitude: float, longitude: float) -> str:
    """
    Retrieves current weather for provided coordinates.
    """
    return f"The current temperature is 15°C with clear sky at coordinates ({latitude}, {longitude})."

@agent_function(model=EmailParams)
def send_email(to: str, body: str) -> str:
    """
    Sends an email to the specified recipient.
    """
    return "Email sent successfully."

@agent_function(model=KBParams)
def search_knowledge_base(query: str, options: dict) -> str:
    """
    Query a knowledge base to retrieve relevant info on a topic.
    """
    results = [
        {"title": "ChatGPT Overview", "snippet": "ChatGPT is a conversational AI model."},
        {"title": "Integrating ChatGPT", "snippet": "ChatGPT can be integrated via API."},
        {"title": "ChatGPT Applications", "snippet": "ChatGPT has diverse applications in many fields."}
    ]
    num_results = options.get("num_results", 3)
    limited_results = results[:num_results]
    return json.dumps(limited_results)

@agent_function(model=AdvancedBookingParams)
def book_flight(flight: dict, passengers: list, seat_preference: Optional[str] = None, meal_preference: Optional[str] = None) -> str:
    """
    Books a flight based on provided details and passenger information.
    """
    # For demonstration, generate a mock booking confirmation.
    booking_id = "BOOK123456"
    confirmation = {
        "booking_id": booking_id,
        "flight": flight,
        "passengers": passengers,
        "seat_preference": seat_preference,
        "meal_preference": meal_preference,
        "status": "confirmed"
    }
    return json.dumps(confirmation)

# -----------------------------------------------------------------------------
# Agent class that simplifies function calling.
# -----------------------------------------------------------------------------
class Agent:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        # Prepare the tools (function schemas) from the registry.
        self.tools = [info["schema"] for info in FUNCTION_REGISTRY.values()]
        logger.debug("Initialized Agent with tools: %s", json.dumps(self.tools, indent=2))

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        logger.debug("Added message: role=%s, content=%s", role, content)

    def run(self, user_message: str) -> str:
        self.add_message("user", user_message)
        logger.debug("Calling OpenAI API with messages: %s", json.dumps(self.messages, indent=2))
        logger.debug("Using tools: %s", json.dumps(self.tools, indent=2))
        
        # First API call to potentially trigger tool calls.
        response = openai.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
            parallel_tool_calls=True
        )
        assistant_message = response.choices[0].message
        logger.debug("Assistant message received: %s", assistant_message)
        self.messages.append(assistant_message)

        # Process any tool calls that the assistant message may include.
        if hasattr(assistant_message, "tool_calls"):
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                logger.debug("Processing tool call for function: %s", func_name)
                try:
                    args = json.loads(tool_call.function.arguments)
                    logger.debug("Parsed arguments for %s: %s", func_name, args)
                except json.JSONDecodeError as e:
                    logger.error("JSON decode error for function %s: %s", func_name, e)
                    args = {}
                func = FUNCTION_REGISTRY[func_name]["function"]
                result = func(**args)
                logger.debug("Result from function %s: %s", func_name, result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
        
        # Second API call to obtain the final response.
        final_response = openai.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools
        )
        final_message = final_response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": final_message})
        logger.debug("Final assistant response: %s", final_message)
        return final_message

# -----------------------------------------------------------------------------
# Example usage of the Agent class with an advanced query.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    agent = Agent(model="gpt-4o")
    query = (
        "I need to know the weather in New York, send an email to alice@example.com saying 'Hello Alice', "
        "search for AI research papers, and finally book a flight from JFK to LAX on 2025-04-01 for two passengers: "
        "John Doe (age 30, passport 'A12345678') and Jane Smith (age 28, passport 'B98765432') with a window seat and a vegetarian meal."
    )
    final_answer = agent.run(query)
    print("Final assistant response:")
    print(final_answer)
