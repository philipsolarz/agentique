from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Union, Literal, Optional, Callable, Type
from rich import print
import jsonref

# -- Recursive JSON Schema model that supports both leaf and nested objects --
class JSONSchemaDefinition(BaseModel):
    type: Union[str, List[str]] = Field(
        ...,
        description="The JSON schema type."
    )
    description: Optional[str] = Field(
        None,
        description="A description of the schema."
    )
    enum: Optional[List[Any]] = Field(
        None,
        description="Optional enumeration of valid values."
    )
    additionalProperties: Optional[bool] = Field(
        None,
        description="Specifies whether additional properties are allowed."
    )
    properties: Optional[Dict[str, "JSONSchemaDefinition"]] = Field(
        None,
        description="A mapping of property names to their schema definitions."
    )
    required: Optional[List[str]] = Field(
        None,
        description="A list of required property names."
    )
    items: Optional["JSONSchemaDefinition"] = Field(
        None,
        description="The schema definition for array items."
    )
    
    class Config:
        extra = "forbid"

# Must update forward references to support recursion.
JSONSchemaDefinition.model_rebuild()

# -- Top-level parameters model is a specialized version of JSONSchemaDefinition --
class JSONSchemaParameters(JSONSchemaDefinition):
    type: Literal["object"] = Field(
        "object",
        description="Must be the string 'object'."
    )
    additionalProperties: bool = Field(
        False,
        description="Indicates whether additional properties are allowed (should be False in strict mode)."
    )
    properties: Dict[str, JSONSchemaDefinition] = Field(
        ...,
        description="A mapping of property names to their schema definitions."
    )
    required: List[str] = Field(
        ...,
        description="A list of required property names."
    )

# -- FunctionDefinition model remains the same --
class FunctionDefinition(BaseModel):
    name: str = Field(
        ...,
        description="The name of the function (e.g. 'get_weather')."
    )
    description: str = Field(
        ...,
        description="A detailed description of when and how to use the function."
    )
    parameters: JSONSchemaParameters = Field(
        ...,
        description="The JSON schema defining the input arguments for the function."
    )
    strict: bool = Field(
        ...,
        description="If true, enforces that function calls adhere strictly to the schema."
    )
    
    class Config:
        extra = "forbid"

# -- Example function and its parameter models --
def get_weather(country: str, location: "LocationParams", forecast_days: List[int]) -> str:
    """Get current weather for a given country, location, and forecast days."""
    return f"It's hot in {location.city}, {country} right now. Forecast for {forecast_days} days."

class CountryEnum(str, Enum):
    Sweden = "Sweden"
    Norway = "Norway"
    Denmark = "Denmark"
    Finland = "Finland"

class LocationParams(BaseModel):
    city: str = Field(..., description="City name.")
    zip_code: Optional[str] = Field(None, description="Optional postal code.")

    class Config:
        extra = "forbid"

class GetWeatherParams(BaseModel):
    country: CountryEnum = Field(..., description="Country to check the weather for.")
    location: LocationParams = Field(..., description="Location details including city and optional postal code.")
    forecast_days: List[int] = Field(
        ...,
        description="List of forecast days to retrieve weather for."
    )

    class Config:
        extra = "forbid"

# -- Helper function to clean extraneous keys from the schema --
def clean_schema(schema: Any) -> Any:
    """
    Recursively remove unwanted keys (like "title", "$defs", "default") and handle "anyOf".
    
    When "anyOf" is encountered, extract the types from each option and replace it with a "type"
    key that is either a string or a list (e.g. ["string", "null"]).
    """
    if isinstance(schema, dict):
        # If "anyOf" is present, process it into a "type" key.
        if "anyOf" in schema:
            types = []
            for option in schema["anyOf"]:
                option_clean = clean_schema(option)
                if "type" in option_clean:
                    t = option_clean["type"]
                    if isinstance(t, list):
                        types.extend(t)
                    else:
                        types.append(t)
            # Remove duplicates while preserving order.
            seen = set()
            unique_types = []
            for t in types:
                if t not in seen:
                    seen.add(t)
                    unique_types.append(t)
            new_dict = {}
            for key, value in schema.items():
                if key in {"title", "$defs", "default", "anyOf"}:
                    continue
                new_dict[key] = clean_schema(value)
            new_dict["type"] = unique_types if len(unique_types) > 1 else unique_types[0]
            return new_dict
        
        cleaned = {}
        for key, value in schema.items():
            if key in {"title", "$defs", "default"}:
                continue
            cleaned_value = clean_schema(value)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value
        return cleaned
    elif isinstance(schema, list):
        return [clean_schema(item) for item in schema if clean_schema(item) is not None]
    else:
        return schema

# -- The transform function --
def transform(fn: Callable, parameter_model: Type[BaseModel]) -> FunctionDefinition:
    """
    Create a FunctionDefinition based on the inner function and its parameter model.
    
    - The function's name is taken from fn.__name__.
    - The description is taken from the function's docstring.
    - The parameters are generated from parameter_model's JSON schema, with additionalProperties set to False.
    """
    name = fn.__name__
    description = fn.__doc__ or f"Function {name}"
    
    # Generate the schema from the parameter model.
    schema = parameter_model.model_json_schema()
    print("Original schema:", schema)

    # Resolve JSON references with merged properties.
    schema = jsonref.replace_refs(schema, merge_props=True)
    print("Reference Replaced:", schema)
    
    # Clean extraneous keys recursively and remove keys with None values.
    schema = clean_schema(schema)
    
    # Enforce strict mode: additionalProperties must be False.
    schema["additionalProperties"] = False
    
    # Convert the cleaned schema into our JSONSchemaParameters model.
    parameters = JSONSchemaParameters.model_validate(schema)
    return FunctionDefinition(name=name, description=description, parameters=parameters, strict=True)

# -- Example usage --
if __name__ == "__main__":
    function_definition = transform(get_weather, GetWeatherParams)
    print("Function definition:")
    print(function_definition.model_dump_json(indent=2, exclude_none=True))
