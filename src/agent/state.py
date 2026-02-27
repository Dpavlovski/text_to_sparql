from typing import TypedDict, Annotated, Any, List

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    attempts: int
    original_question: str
    language: str
    ner_keywords: List[dict]
    candidates: str
    schema_context: str
    examples: str
    log_data: Annotated[list[dict[str, Any]], lambda x, y: x + y]