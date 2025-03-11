"""
Schema utilities for OpenAI function calling in the Agentique library.

Provides utilities to generate accurate OpenAI function schemas from 
Python functions and their associated Pydantic parameter models.
"""

from enum import Enum
from typing import Any, Dict, List, Union, Literal, Optional, Callable, Type
import jsonref
from pydantic import BaseModel

from .logging import get_logger

logger = get_logger("schema")

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
    in_progress: Optional[set] = None
) -> Any:
    """
    Main entry point for normalizing the JSON schema node.
    - in_progress tracks nodes we are *currently* processing to avoid cyc references.
    - memo stores *finished* nodes to avoid reprocessing them multiple times.
    """
    if memo is None:
        memo = {}
    if in_progress is None:
        in_progress = set()
    
    node_id = id(node)
    
    # If we have a finished node in memo, return it
    if node_id in memo:
        return memo[node_id]
    
    # If we're *already* processing this node => cyc reference
    if node_id in in_progress:
        # Return a minimal structure to short-circuit
        empty_dict: Dict[str, Any] = {}
        memo[node_id] = empty_dict
        return empty_dict
    
    # Mark that we are processing this node
    in_progress.add(node_id)
    
    # Process based on node type
    if isinstance(node, dict):
        processed = process_dict_node(node, memo, in_progress)
        memo[node_id] = processed
        in_progress.remove(node_id)
        return processed
    
    elif isinstance(node, list):
        processed = process_list_node(node, memo, in_progress)
        memo[node_id] = processed
        in_progress.remove(node_id)
        return processed
    
    else:
        # Scalar node => just store it in memo
        memo[node_id] = node
        in_progress.remove(node_id)
        return node

# -------------------------------
# Dict Node Handler
# -------------------------------

def process_dict_node(schema_dict: Dict[str, Any], memo: Dict[int, Any], in_progress: set) -> Dict[str, Any]:
    """
    Handle a dictionary node in the schema.
    1. If "anyOf" is present => unify it
    2. Otherwise:
       - remove unwanted keys
       - recursively process each child
       - enforce object constraints if 'type' == 'object'
    """
    if "anyOf" in schema_dict:
        return unify_anyof(schema_dict, memo, in_progress)
    
    # Remove unwanted keys
    pruned = remove_unwanted(schema_dict)
    
    # Recur on children
    result: Dict[str, Any] = {}
    for k, v in pruned.items():
        result[k] = process_schema(v, memo, in_progress)
    
    # If 'type' is 'object', force additionalProperties=false
    enforce_object_constraints(result)
    return result

# -------------------------------
# List Node Handler
# -------------------------------

def process_list_node(schema_list: List[Any], memo: Dict[int, Any], in_progress: set) -> List[Any]:
    """
    Process each element in a list by normalizing it.
    """
    new_list = []
    for item in schema_list:
        processed_item = process_schema(item, memo, in_progress)
        if processed_item is not None:
            new_list.append(processed_item)
    return new_list

# -------------------------------
# Union Handler
# -------------------------------

def unify_anyof(schema_dict: Dict[str, Any], memo: Dict[int, Any], in_progress: set) -> Dict[str, Any]:
    """
    Merge an 'anyOf' union into a single node by collecting all type options.
    If exactly one option is an object (with 'properties') or array (with 'items'),
    merge that option's details and unify the 'type' with null. This is how
    we handle optional fields typed e.g. Optional[MyObject] or Optional[List[Foo]].
    
    Otherwise, produce a node with 'type': union_of_all_types.
    """
    # Gather the raw union options
    raw_options = schema_dict["anyOf"]
    processed_opts = [process_schema(opt, memo, in_progress) for opt in raw_options]
    
    # Collect all "type" values
    union_types: List[Union[str, List[str]]] = []
    for opt in processed_opts:
        t = opt.get("type")
        if isinstance(t, list):
            union_types.extend(t)
        elif isinstance(t, str):
            union_types.append(t)
    
    # De-duplicate while preserving order
    seen_types = []
    for t in union_types:
        if t not in seen_types:
            seen_types.append(t)
    
    # Identify object-likes or array-likes
    object_opts = [o for o in processed_opts if o.get("type") == "object" and "properties" in o]
    array_opts = [o for o in processed_opts if o.get("type") == "array" and "items" in o]
    
    # We'll handle the common scenario: union of [object, null] or [array, null]
    # => unify the object or array with "type": ["object","null"] or ["array","null"].
    
    # Check for exactly one object-likes and all others are "null"
    if len(object_opts) == 1 and all(is_null_type(opt) for opt in processed_opts if opt not in object_opts):
        # Merge that single object option
        single_obj = object_opts[0].copy()
        new_type = list(set(seen_types))
        single_obj["type"] = new_type if len(new_type) > 1 else new_type[0]
        
        # Merge additional keys from parent
        pruned = remove_unwanted(schema_dict, ignore_anyof=True)
        for k, v in pruned.items():
            single_obj[k] = process_schema(v, memo, in_progress)
        
        enforce_object_constraints(single_obj)
        return single_obj
    
    # Check for exactly one array-likes and all others are "null"
    if len(array_opts) == 1 and all(is_null_type(opt) for opt in processed_opts if opt not in array_opts):
        single_arr = array_opts[0].copy()
        new_type = list(set(seen_types))
        single_arr["type"] = new_type if len(new_type) > 1 else new_type[0]
        
        # Merge additional keys from parent
        pruned = remove_unwanted(schema_dict, ignore_anyof=True)
        for k, v in pruned.items():
            single_arr[k] = process_schema(v, memo, in_progress)
        
        enforce_object_constraints(single_arr)  # If it's array+null, no big harm
        return single_arr
    
    # Otherwise, produce a final node with union of all types
    final: Dict[str, Any] = {}
    pruned = remove_unwanted(schema_dict, ignore_anyof=True)
    for k, v in pruned.items():
        final[k] = process_schema(v, memo, in_progress)
    final["type"] = seen_types if len(seen_types) > 1 else seen_types[0]
    return final

def is_null_type(opt: Dict[str, Any]) -> bool:
    """
    Return True if this schema node is "type": "null" or
    "type" is a list containing "null".
    """
    t = opt.get("type")
    if isinstance(t, list):
        return "null" in t and len(t) == 1
    return t == "null"

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

def generate_function_schema(fn: Callable, parameter_model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Generate an OpenAI function schema from a function and its Pydantic parameter model.
    
    Args:
        fn: The function for which to generate a schema
        parameter_model: Pydantic model defining the function parameters
        
    Returns:
        OpenAI function definition schema
    """
    name = fn.__name__
    description = fn.__doc__ or f"Function {name}"
    
    # Generate the raw schema from the parameter model
    raw_schema = parameter_model.model_json_schema()
    
    # Merge JSON references
    resolved_schema = jsonref.replace_refs(raw_schema, merge_props=True)
    
    # Process the schema
    final_schema = process_schema(resolved_schema)
    
    # At the top level, ensure additionalProperties is false
    final_schema["additionalProperties"] = False
    
    # Create function definition
    function_def = {
        "name": name,
        "description": description,
        "parameters": final_schema
    }
    
    logger.debug(f"Generated function schema for: [bold blue]{name}[/bold blue]")
    return function_def