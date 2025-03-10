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

# Update forward references for recursion.
JSONSchemaDefinition.model_rebuild()

# -- Top-level parameters model (specialized version) --
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

# -- FunctionDefinition model remains unchanged --
class FunctionDefinition(BaseModel):
    name: str = Field(
        ...,
        description="The name of the function (e.g. 'advanced_analysis')."
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

# -- Advanced parameter models for testing extremes --

# Enums
class StatusEnum(str, Enum):
    active = "active"
    inactive = "inactive"
    pending = "pending"

class LevelEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

# A nested model for detailed item information.
class Detail(BaseModel):
    description: str = Field(..., description="Detailed description of the item.")
    parameters: Dict[str, Union[str, int, float]] = Field(
        ..., description="Arbitrary parameters related to the item."
    )
    
    class Config:
        extra = "forbid"

# An item model nested inside a list.
class Item(BaseModel):
    name: str = Field(..., description="Name of the item.")
    value: Union[int, float] = Field(..., description="Numerical value for the item.")
    flags: List[bool] = Field(..., description="List of boolean flags.")
    detail: Detail = Field(..., description="Detailed information for the item.")
    
    class Config:
        extra = "forbid"

# A recursive tree node model.
class TreeNode(BaseModel):
    node_id: str = Field(..., description="Unique identifier for the node.")
    children: Optional[List["TreeNode"]] = Field(
        None, description="Optional list of child nodes."
    )
    
    class Config:
        extra = "forbid"

TreeNode.model_rebuild()  # For recursion

# A model for a config that can either be a string or an object.
class ConfigObject(BaseModel):
    mode: str = Field(..., description="Mode for configuration.")
    options: Dict[str, Any] = Field(..., description="Additional configuration options.")
    
    class Config:
        extra = "forbid"

# Advanced top-level parameters.
class AdvancedAnalysisParams(BaseModel):
    id: str = Field(..., description="Unique identifier for the analysis.")
    title: str = Field(..., description="Title of the analysis.")
    status: StatusEnum = Field(..., description="Current status of the analysis.")
    metadata: Optional[Dict[str, Union[str, int, bool]]] = Field(
        None, description="Optional metadata."
    )
    tags: Optional[List[str]] = Field(None, description="List of tags for the analysis.")
    items: List[Item] = Field(..., description="List of items to analyze.")
    tree: Optional[List[TreeNode]] = Field(None, description="A recursive tree of nodes.")
    config: Union[str, ConfigObject] = Field(
        ..., description="Configuration as a simple string or a detailed object."
    )
    levels: List[LevelEnum] = Field(..., description="List of level indicators.")
    
    class Config:
        extra = "forbid"

# -- The advanced_analysis function --
def advanced_analysis(params: AdvancedAnalysisParams) -> str:
    """
    Performs an advanced analysis based on the provided parameters.
    
    This function demonstrates the extremes of our function calling schema,
    including nested objects, arrays, unions, enums, and recursive structures.
    """
    return f"Analysis {params.id} with title '{params.title}' is {params.status}."

# -- Updated clean_schema: remove unwanted keys and convert anyOf unions into a merged "type" --
def clean_schema(schema: Any, memo: Optional[Dict[int, Any]] = None) -> Any:
    """
    Recursively remove unwanted keys ("title", "$defs", "default") and handle anyOf.
    
    For anyOf, we assume that the union represents a simple union (commonly a type union
    like object vs. null or integer vs. number). We collect all types from the options.
    If one of the options is an object (has "properties"), we merge its details.
    Otherwise, we simply return a dictionary with a "type" key whose value is a union list.
    
    Memoization is used to prevent infinite recursion on cyclic structures.
    Additionally, for any node representing an object, we force "additionalProperties": false.
    """
    if memo is None:
        memo = {}
    schema_id = id(schema)
    if schema_id in memo:
        return memo[schema_id]
    
    if isinstance(schema, dict):
        # Handle "anyOf" if present.
        if "anyOf" in schema:
            options = [clean_schema(option, memo) for option in schema["anyOf"]]
            union_types = []
            for opt in options:
                if "type" in opt:
                    t = opt["type"]
                    if isinstance(t, list):
                        union_types.extend(t)
                    else:
                        union_types.append(t)
            # Remove duplicates while preserving order.
            unique_types = []
            for t in union_types:
                if t not in unique_types:
                    unique_types.append(t)
            # Check if any option is an object with properties.
            object_opts = [opt for opt in options if opt.get("type") == "object" and "properties" in opt]
            if object_opts:
                # Use the first object option as the basis.
                basis = object_opts[0].copy()
                basis["type"] = unique_types if len(unique_types) > 1 else unique_types[0]
                # Copy over any keys from the parent (except unwanted ones).
                for key, value in schema.items():
                    if key in {"title", "$defs", "default", "anyOf"}:
                        continue
                    basis[key] = clean_schema(value, memo)
                if basis.get("type") == "object":
                    basis["additionalProperties"] = False
                memo[schema_id] = basis
                return basis
            else:
                new_dict = {}
                for key, value in schema.items():
                    if key in {"title", "$defs", "default", "anyOf"}:
                        continue
                    new_dict[key] = clean_schema(value, memo)
                new_dict["type"] = unique_types if len(unique_types) > 1 else unique_types[0]
                memo[schema_id] = new_dict
                return new_dict
        
        new_dict: Dict[str, Any] = {}
        memo[schema_id] = new_dict
        for key, value in schema.items():
            if key in {"title", "$defs", "default"}:
                continue
            new_dict[key] = clean_schema(value, memo)
        if new_dict.get("type") == "object":
            new_dict["additionalProperties"] = False
        memo[schema_id] = new_dict
        return new_dict

    elif isinstance(schema, list):
        new_list = [clean_schema(item, memo) for item in schema if clean_schema(item, memo) is not None]
        memo[schema_id] = new_list
        return new_list
    else:
        return schema

# -- The transform function remains unchanged --
def transform(fn: Callable, parameter_model: Type[BaseModel]) -> FunctionDefinition:
    """
    Create a FunctionDefinition based on the inner function and its parameter model.
    
    - The function's name is taken from fn.__name__.
    - The description is taken from the function's docstring.
    - The parameters are generated from parameter_model's JSON schema, with additionalProperties set to False.
    """
    name = fn.__name__
    description = fn.__doc__ or f"Function {name}"
    
    schema = parameter_model.model_json_schema()
    print("Original schema:", schema)
    
    # Resolve JSON references with merged properties.
    schema = jsonref.replace_refs(schema, merge_props=True)
    print("Reference Replaced:", schema)
    
    schema = clean_schema(schema)
    schema["additionalProperties"] = False
    
    parameters = JSONSchemaParameters.model_validate(schema)
    return FunctionDefinition(name=name, description=description, parameters=parameters, strict=True)

# -- Example usage --
if __name__ == "__main__":
    function_definition = transform(advanced_analysis, AdvancedAnalysisParams)
    print("Advanced Function definition:")
    print(function_definition.model_dump_json(indent=2, exclude_none=True))
