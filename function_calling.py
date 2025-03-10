import openai
import json
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Define functions that our assistant can call.
# In a real implementation these might perform real API calls.
# ---------------------------------------------------------------------------

def get_weather(latitude, longitude):
    """
    Simulates retrieving current weather for provided coordinates.
    In production, this might call an external weather API.
    """
    # Hard-coded weather data for demonstration.
    weather_data = {
        "temperature": 15,  # Example temperature in Celsius.
        "description": "Clear sky"
    }
    return f"The current temperature is {weather_data['temperature']}°C with {weather_data['description']}."

def send_email(to, body):
    """
    Simulates sending an email.
    In production, this function could integrate with an email service.
    """
    # Simulate sending the email.
    return "success"

def search_knowledge_base(query, options):
    """
    Simulates querying a knowledge base to retrieve relevant info on a topic.
    In production, this function might perform a search query on an external data source.
    """
    # Hard-coded search results for demonstration.
    results = [
        {"title": "ChatGPT Overview", "snippet": "ChatGPT is a conversational AI model."},
        {"title": "Integrating ChatGPT", "snippet": "ChatGPT can be integrated via API."},
        {"title": "ChatGPT Applications", "snippet": "ChatGPT has diverse applications in many fields."}
    ]
    num_results = options.get("num_results", 3)
    limited_results = results[:num_results]
    return json.dumps(limited_results)

# ---------------------------------------------------------------------------
# Define function schemas with strict mode enabled.
# The schemas follow JSON Schema with additionalProperties set to False.
# ---------------------------------------------------------------------------

get_weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Retrieves current weather for provided coordinates.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude of the location."},
                "longitude": {"type": "number", "description": "Longitude of the location."}
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        }
    }
}

send_email_schema = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Sends an email to the specified recipient.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "body": {"type": "string", "description": "Content of the email."}
            },
            "required": ["to", "body"],
            "additionalProperties": False
        }
    }
}

search_kb_schema = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Query a knowledge base to retrieve relevant info on a topic.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user question or search query."
                },
                "options": {
                    "type": "object",
                    "properties": {
                        "num_results": {
                            "type": "number",
                            "description": "Number of top results to return."
                        },
                        "domain_filter": {
                            "type": ["string", "null"],
                            "description": "Optional domain to narrow the search (e.g. 'finance', 'medical'). Pass null if not needed."
                        },
                        "sort_by": {
                            "type": ["string", "null"],
                            "enum": ["relevance", "date", "popularity", "alphabetical"],
                            "description": "How to sort results. Pass null if not needed."
                        }
                    },
                    "required": ["num_results", "domain_filter", "sort_by"],
                    "additionalProperties": False
                }
            },
            "required": ["query", "options"],
            "additionalProperties": False
        }
    }
}

# ---------------------------------------------------------------------------
# Assemble all available tools (functions) into a list.
# ---------------------------------------------------------------------------
tools = [get_weather_schema, send_email_schema, search_kb_schema]

# ---------------------------------------------------------------------------
# Create the initial conversation messages.
# The system message sets the assistant's context.
# ---------------------------------------------------------------------------
messages = [
    {"role": "system", "content": "You are an assistant that can fetch data and perform actions like sending emails."},
    {"role": "user", "content": "Can you tell me the weather in Paris and Bogotá? Also, send an email to bob@example.com saying 'Hi Bob', and search the AI knowledge base for ChatGPT information."}
]

# ---------------------------------------------------------------------------
# Call the ChatCompletion endpoint with function calling enabled.
# Enable parallel_tool_calls to allow multiple function calls in one turn.
# tool_choice is set to "auto" so the model can decide which functions to call.
# ---------------------------------------------------------------------------
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",       # Model decides which tools to use.
    parallel_tool_calls=True  # Allow multiple function calls in a single turn.
)

# ---------------------------------------------------------------------------
# Retrieve the assistant's message containing tool_calls and append it.
# ---------------------------------------------------------------------------
assistant_message = response.choices[0].message
messages.append(assistant_message)

# ---------------------------------------------------------------------------
# Print the initial tool calls returned by the assistant.
# ---------------------------------------------------------------------------
tool_calls = assistant_message.tool_calls
print("Initial tool calls from the assistant:")
print(tool_calls)

# ---------------------------------------------------------------------------
# Process each tool call by executing the corresponding function.
# Append each tool response immediately after the assistant's message.
# ---------------------------------------------------------------------------
for tool_call in tool_calls:
    call_id = tool_call.id
    function_info = tool_call.function
    function_name = function_info.name
    try:
        arguments = json.loads(function_info.arguments)
    except json.JSONDecodeError:
        arguments = {}

    # Execute the function based on its name.
    if function_name == "get_weather":
        # Expecting 'latitude' and 'longitude' in arguments.
        result = get_weather(arguments["latitude"], arguments["longitude"])
    elif function_name == "send_email":
        result = send_email(**arguments)
    elif function_name == "search_knowledge_base":
        result = search_knowledge_base(**arguments)
    else:
        result = f"Error: Function '{function_name}' is not implemented."
    
    # Append the tool's response to the conversation.
    messages.append({
        "role": "tool",
        "tool_call_id": call_id,
        "content": str(result)
    })

# ---------------------------------------------------------------------------
# Call the model again with the updated conversation including tool results.
# The model can now incorporate the function call outputs into its final response.
# ---------------------------------------------------------------------------
final_response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)

# ---------------------------------------------------------------------------
# Print the final assistant response.
# ---------------------------------------------------------------------------
print("\nFinal assistant response:")
print(final_response.choices[0].message.content)
