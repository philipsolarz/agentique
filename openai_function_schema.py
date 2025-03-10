from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Union, Literal, Optional, Callable, Type
import jsonref

# -------------------------------
# Schema Models for Function Calling
# -------------------------------

class FunctionSchemaDefinition(BaseModel):
    """Represents a JSON schema definition for a function parameter."""
    type: Union[str, List[str]] = Field(..., description="The JSON schema type.")
    description: Optional[str] = Field(None, description="A description of the schema.")
    enum: Optional[List[Any]] = Field(None, description="Optional enumeration of valid values.")
    additionalProperties: Optional[bool] = Field(None, description="Specifies whether additional properties are allowed.")
    properties: Optional[Dict[str, "FunctionSchemaDefinition"]] = Field(
        None, description="Mapping of property names to their schema definitions."
    )
    required: Optional[List[str]] = Field(None, description="List of required property names.")
    items: Optional["FunctionSchemaDefinition"] = Field(None, description="Schema definition for array items.")
    
    class Config:
        extra = "forbid"

FunctionSchemaDefinition.model_rebuild()

class FunctionSchemaParameters(FunctionSchemaDefinition):
    """Specialized schema for function parameters (an object with properties)."""
    type: Literal["object"] = Field("object", description="Must be 'object'.")
    additionalProperties: bool = Field(False, description="Should be false in strict mode.")
    properties: Dict[str, FunctionSchemaDefinition] = Field(
        ..., description="Mapping of parameter names to their schema definitions."
    )
    required: List[str] = Field(..., description="List of required parameter names.")

class OpenAIFunctionDefinition(BaseModel):
    """Represents an OpenAI function definition for function calling."""
    name: str = Field(..., description="The function's name.")
    description: str = Field(..., description="Description of the function.")
    parameters: FunctionSchemaParameters = Field(
        ..., description="The JSON schema defining the function's input arguments."
    )
    strict: bool = Field(..., description="If true, the function call must strictly follow the schema.")
    
    class Config:
        extra = "forbid"


# -------------------------------
# Config / Constants
# -------------------------------

UNWANTED_KEYS = {
    "title", 
    "$defs", 
    "default",
    "maximum",
    "minimum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "pattern",
    "format",
    "maxLength",
    "minLength",
    "const",
}

# -------------------------------
# Primary Dispatch
# -------------------------------

def process_schema(
    node: Any,
    memo: Optional[Dict[int, Any]] = None,
    inProgress: Optional[set] = None
) -> Any:
    """
    Main entry point for normalizing the JSON schema node.
    - inProgress tracks nodes we are *currently* processing to avoid cyc references.
    - memo stores *finished* nodes to avoid reprocessing them multiple times.
    """
    if memo is None:
        memo = {}
    if inProgress is None:
        inProgress = set()
    
    node_id = id(node)
    
    # If we have a finished node in memo, return immediately
    if node_id in memo:
        return memo[node_id]
    
    # If we're *already* in the middle of processing this node => cyc reference
    if node_id in inProgress:
        # Return a minimal structure, e.g. empty dict or None
        # so we don't keep recursing infinitely.
        empty_dict: Dict[str, Any] = {}
        memo[node_id] = empty_dict
        return empty_dict
    
    # Mark that we are processing this node
    inProgress.add(node_id)
    
    # Process based on node type
    if isinstance(node, dict):
        processed = process_dict_node(node, memo, inProgress)
        memo[node_id] = processed
        inProgress.remove(node_id)
        return processed
    
    elif isinstance(node, list):
        processed = process_list_node(node, memo, inProgress)
        memo[node_id] = processed
        inProgress.remove(node_id)
        return processed
    
    else:
        # Scalar node => just store it in memo
        memo[node_id] = node
        inProgress.remove(node_id)
        return node

# -------------------------------
# Dict Node Handler
# -------------------------------

def process_dict_node(schema_dict: Dict[str, Any], memo: Dict[int, Any], inProgress: set) -> Dict[str, Any]:
    """
    Handle a dictionary node in the schema.
    1. If "anyOf" is present => unify it
    2. Otherwise:
       - remove unwanted keys
       - recursively process each child
       - enforce object constraints if 'type' == 'object'
    """
    if "anyOf" in schema_dict:
        return unify_anyof(schema_dict, memo, inProgress)
    
    # Remove unwanted keys
    pruned = remove_unwanted(schema_dict)
    
    # Recur on children
    result: Dict[str, Any] = {}
    for k, v in pruned.items():
        result[k] = process_schema(v, memo, inProgress)
    
    # If 'type' is 'object', force additionalProperties=false
    enforce_object_constraints(result)
    return result

# -------------------------------
# List Node Handler
# -------------------------------

def process_list_node(schema_list: List[Any], memo: Dict[int, Any], inProgress: set) -> List[Any]:
    """
    Process each element in a list by normalizing it.
    """
    new_list = []
    for item in schema_list:
        processed_item = process_schema(item, memo, inProgress)
        if processed_item is not None:
            new_list.append(processed_item)
    return new_list

# -------------------------------
# Union Handler
# -------------------------------

def unify_anyof(schema_dict: Dict[str, Any], memo: Dict[int, Any], inProgress: set) -> Dict[str, Any]:
    """
    Merge an 'anyOf' union into a single node by collecting all type options.
    If one option is an object with 'properties', merge that option's details.
    Otherwise, produce a final node with 'type' set to the union of all types.
    """
    # Gather the raw union options
    options = schema_dict["anyOf"]
    processed_opts = [process_schema(opt, memo, inProgress) for opt in options]
    
    # Collect all type hints
    union_types: List[str] = []
    for opt in processed_opts:
        t = opt.get("type")
        if isinstance(t, list):
            union_types.extend(t)
        elif isinstance(t, str):
            union_types.append(t)
    # Remove duplicates preserving order
    unique_types: List[str] = []
    for t in union_types:
        if t not in unique_types:
            unique_types.append(t)
    
    # See if we have an object option
    object_opts = [o for o in processed_opts if o.get("type") == "object" and "properties" in o]
    if object_opts:
        # Start from the first object-based option
        basis = object_opts[0].copy()
        basis["type"] = unique_types if len(unique_types) > 1 else unique_types[0]
        
        # Merge other keys from the parent except "anyOf" + unwanted
        # so we can preserve e.g. 'description' if it existed
        pruned = remove_unwanted(schema_dict, ignore_anyof=True)
        for k, v in pruned.items():
            # If we didn't handle it, process recursively
            basis[k] = process_schema(v, memo, inProgress)
        
        enforce_object_constraints(basis)
        return basis
    else:
        # Just produce a final node with union of types
        final: Dict[str, Any] = {}
        pruned = remove_unwanted(schema_dict, ignore_anyof=True)
        for k, v in pruned.items():
            final[k] = process_schema(v, memo, inProgress)
        final["type"] = unique_types if len(unique_types) > 1 else unique_types[0]
        return final

# -------------------------------
# Utility
# -------------------------------

def remove_unwanted(schema_dict: Dict[str, Any], ignore_anyof: bool = False) -> Dict[str, Any]:
    """
    Return a shallow copy of schema_dict omitting the UNWANTED_KEYS.
    If ignore_anyof = True, also remove 'anyOf'.
    """
    unwanted_keys = set(UNWANTED_KEYS)
    if ignore_anyof:
        unwanted_keys.add("anyOf")
    return {k: v for k, v in schema_dict.items() if k not in unwanted_keys}

def enforce_object_constraints(schema_dict: Dict[str, Any]) -> None:
    """
    If 'type' is 'object', set additionalProperties=False.
    """
    if schema_dict.get("type") == "object":
        schema_dict["additionalProperties"] = False

# -------------------------------
# Public API
# -------------------------------

def generate_function_schema(fn: Callable, parameter_model: Type[BaseModel]) -> OpenAIFunctionDefinition:
    """
    Generate an OpenAI function schema from a function and its Pydantic parameter model.
    
    Steps:
      1. Extract function name and docstring.
      2. Generate the JSON schema from the parameter model.
      3. Resolve JSON references (merging properties).
      4. Process the schema (removing unwanted keys, merging union types, etc.).
      5. Validate against our FunctionSchemaParameters model.
    """
    name = fn.__name__
    description = fn.__doc__ or f"Function {name}"
    
    # 1-2. Generate the raw schema from the parameter model
    raw_schema = parameter_model.model_json_schema()
    
    # 3. Merge JSON references
    resolved_schema = jsonref.replace_refs(raw_schema, merge_props=True)
    
    # 4. Process the schema
    final_schema = process_schema(resolved_schema)
    
    # At the top level, ensure additionalProperties is false
    final_schema["additionalProperties"] = False
    
    # 5. Validate
    parameters = FunctionSchemaParameters.model_validate(final_schema)
    
    return OpenAIFunctionDefinition(
        name=name,
        description=description,
        parameters=parameters,
        strict=True
    )
