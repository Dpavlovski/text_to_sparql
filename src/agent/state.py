import operator
from typing import Annotated, Any, Dict, List, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    The state of the SPARQL Agent during execution.
    Annotated fields use operators to tell LangGraph how to update the state.
    """
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)

    attempts: int = Field(default=0)
    original_question: str
    language: str = Field(default="en")

    mode: Literal["zero_shot", "full"] = Field(default="full")

    ner_keywords: List[Dict[str, Any]] = Field(default_factory=list)
    candidates: str = Field(default="")
    schema_context: str = Field(default="")
    examples: str = Field(default="")
    current_results: Any = Field(default=None)

    log_data: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)
