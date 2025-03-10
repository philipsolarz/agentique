from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Union, Literal, Optional
from rich import print
from openai_function_schema import generate_function_schema, OpenAIFunctionDefinition

# -------------------------------
# Advanced Parameter Models for Testing (with constraints)
# -------------------------------

class StatusEnum(str, Enum):
    active = "active"
    inactive = "inactive"
    pending = "pending"

class LevelEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class Detail(BaseModel):
    description: str = Field(..., description="Detailed description of the item.")
    parameters: Dict[str, Union[str, int, float]] = Field(
        ..., description="Arbitrary parameters related to the item."
    )
    rating: float = Field(..., ge=0, le=5, description="Rating between 0 and 5.")
    class Config:
        extra = "forbid"

class Item(BaseModel):
    name: str = Field(..., description="Name of the item.")
    value: Union[int, float] = Field(..., description="Numerical value for the item.")
    flags: List[bool] = Field(..., description="List of boolean flags.")
    detail: Detail = Field(..., description="Detailed information for the item.")
    quantity: int = Field(..., gt=0, description="Quantity must be greater than 0.")
    class Config:
        extra = "forbid"

class TreeNode(BaseModel):
    node_id: str = Field(..., description="Unique identifier for the node.")
    children: Optional[List["TreeNode"]] = Field(None, description="Optional list of child nodes.")
    class Config:
        extra = "forbid"
TreeNode.model_rebuild()

class ConfigObject(BaseModel):
    mode: str = Field(..., description="Mode for configuration.")
    options: Dict[str, Any] = Field(..., description="Additional configuration options.")
    class Config:
        extra = "forbid"

class AdvancedAnalysisParams(BaseModel):
    id: str = Field(..., description="Unique identifier for the analysis.")
    title: str = Field(..., description="Title of the analysis.")
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
    class Config:
        extra = "forbid"

def advanced_analysis(params: AdvancedAnalysisParams) -> str:
    """
    Performs an advanced analysis based on the provided parameters.
    
    This function demonstrates the extremes of our function calling schema,
    including nested objects, arrays, unions, enums, numeric constraints, and recursive structures.
    """
    return f"Analysis {params.id} with title '{params.title}' is {params.status}."

# -------------------------------
# Generate and Output the Function Schema
# -------------------------------
if __name__ == "__main__":
    function_schema: OpenAIFunctionDefinition = generate_function_schema(advanced_analysis, AdvancedAnalysisParams)
    print("Advanced Function Schema:")
    print(function_schema.model_dump_json(indent=2, exclude_none=True))
