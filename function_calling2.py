import openai
import json
import logging
from typing import Callable, Dict, Any, List, Type, Optional
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Set up debug logging.
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global registry to hold functions registered as tools.
# -----------------------------------------------------------------------------
FUNCTION_REGISTRY: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Helper function to recursively enforce "additionalProperties": False on all objects.
# This version traverses all keys including "$defs".
# -----------------------------------------------------------------------------
def enforce_additional_properties(schema: dict) -> dict:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            logger.debug("Set additionalProperties=False for schema: %s", schema.get("title", "object"))
            if "properties" in schema:
                for key, prop in schema["properties"].items():
                    schema["properties"][key] = enforce_additional_properties(prop)
        if "items" in schema:
            schema["items"] = enforce_additional_properties(schema["items"])
        if "$defs" in schema:
            for def_key, def_value in schema["$defs"].items():
                schema["$defs"][def_key] = enforce_additional_properties(def_value)
        for key, value in schema.items():
            if isinstance(value, dict) and key not in ("properties", "items", "$defs"):
                schema[key] = enforce_additional_properties(value)
        for key, value in schema.items():
            if isinstance(value, list):
                schema[key] = [enforce_additional_properties(item) if isinstance(item, dict) else item for item in value]
    return schema

# -----------------------------------------------------------------------------
# Decorator to register a function as an agent function with an associated Pydantic model.
# Automatically generates a JSON schema for the function's parameters using the model.
# -----------------------------------------------------------------------------
def agent_function(model: Type[BaseModel]):
    def decorator(func: Callable):
        # Get the function description from the docstring.
        func_description = (func.__doc__ or "No description provided.").strip()
        # Generate JSON schema using Pydantic's model_json_schema (v2).
        schema = model.model_json_schema()
        logger.debug("Original schema for %s: %s", func.__name__, json.dumps(schema, indent=2))
        # Enforce additionalProperties: False recursively.
        schema = enforce_additional_properties(schema)
        logger.debug("Enforced schema for %s: %s", func.__name__, json.dumps(schema, indent=2))

        # Create the function schema as expected by the OpenAI API.
        function_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func_description,
                "strict": True,
                "parameters": schema,
            }
        }
        # Register the function in the global registry.
        FUNCTION_REGISTRY[func.__name__] = {
            "function": func,
            "schema": function_schema,
            "model": model,
        }
        logger.debug("Registered function '%s' with schema: %s", func.__name__, json.dumps(function_schema, indent=2))
        return func
    return decorator

# -----------------------------------------------------------------------------
# Example Pydantic models for function parameters.
# Use Optional[...] to allow null values and produce union types.
# -----------------------------------------------------------------------------
class WeatherParams(BaseModel):
    latitude: float = Field(..., description="Latitude of the location.")
    longitude: float = Field(..., description="Longitude of the location.")

class EmailParams(BaseModel):
    to: str = Field(..., description="Recipient email address.")
    body: str = Field(..., description="Content of the email.")

# Define an enum for sort options.
class SortOptions(str, Enum):
    relevance = "relevance"
    date = "date"
    popularity = "popularity"
    alphabetical = "alphabetical"

class KBOptions(BaseModel):
    num_results: int = Field(..., description="Number of top results to return.")
    # Mark as Optional[...] so that the JSON schema type becomes ["string", "null"]
    domain_filter: Optional[str] = Field(..., description="Optional domain to narrow the search (e.g. 'finance', 'medical'). Pass null if not needed.")
    sort_by: SortOptions = Field(..., description="How to sort results. Pass null if not needed.")

class KBParams(BaseModel):
    query: str = Field(..., description="The user question or search query.")
    options: KBOptions

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

# -----------------------------------------------------------------------------
# Agent class that simplifies function calling.
# It collects registered functions, handles calling the OpenAI API,
# and dispatches tool calls to the corresponding functions.
# -----------------------------------------------------------------------------
class Agent:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        # Collect tools from the global FUNCTION_REGISTRY.
        self.tools = [info["schema"] for info in FUNCTION_REGISTRY.values()]
        logger.debug("Initialized Agent with tools: %s", json.dumps(self.tools, indent=2))

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        logger.debug("Added message: role=%s, content=%s", role, content)

    def run(self, user_message: str) -> str:
        self.add_message("user", user_message)
        logger.debug("Calling OpenAI API with messages: %s", json.dumps(self.messages, indent=2))
        logger.debug("Using tools: %s", json.dumps(self.tools, indent=2))
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

                model_cls = FUNCTION_REGISTRY[func_name]["model"]
                try:
                    parsed_args = model_cls.parse_obj(args).dict()
                    logger.debug("Validated arguments for %s: %s", func_name, parsed_args)
                except Exception as e:
                    logger.error("Validation error for function %s: %s", func_name, e)
                    parsed_args = args

                func = FUNCTION_REGISTRY[func_name]["function"]
                result = func(**parsed_args)
                logger.debug("Result from function %s: %s", func_name, result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

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
# Example usage of the Agent class.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    agent = Agent(model="gpt-4o")
    query = (
        "Can you tell me the weather in Paris and Bogotá? "
        "Also, send an email to bob@example.com saying 'Hi Bob', "
        "and search the AI knowledge base for ChatGPT information."
    )
    final_answer = agent.run(query)
    print("Final assistant response:")
    print(final_answer)
