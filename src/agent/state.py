from typing import TypedDict, Annotated, Any

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    attempts: int
    original_question: str
    log_data: Annotated[list[dict[str, Any]], lambda x, y: x + y]
