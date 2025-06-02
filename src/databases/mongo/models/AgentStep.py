from typing import Any, Dict

from bson import ObjectId
from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    run_id: str = Field(..., description="Unique identifier for the processing run")
    question: str = Field(..., description="Original user question")
    step_number: int = Field(..., description="Sequential step number")
    tool_name: str = Field(..., description="Name of the tool used")
    tool_input: Dict[str, Any] = Field(..., description="Input parameters passed to the tool")
    tool_output: Dict[str, Any] | str = Field(..., description="Output returned by the tool")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

    @classmethod
    def serialize_value(cls, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, dict):
            return {k: cls.serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, set)):
            return [cls.serialize_value(item) for item in value]
        elif hasattr(value, "model_dump"):
            return value.model_dump()
        elif hasattr(value, "__dict__"):
            return vars(value)
        else:
            return str(value)
