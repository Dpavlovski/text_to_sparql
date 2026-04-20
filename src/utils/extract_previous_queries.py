import re
from typing import List

from langchain_core.messages import ToolMessage

from src.agent.state import AgentState


async def extract_previous_queries(state: AgentState) -> List[str]:
    sparql_pattern = re.compile(
        r"""\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b.*?(?<=})""",
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )

    previous_queries = []
    # Because state is a Pydantic BaseModel, dot notation works perfectly:
    for msg in state.messages:
        if isinstance(msg, ToolMessage):
            match = sparql_pattern.search(msg.content)
            if match:
                previous_queries.append(match.group(0).strip())

    return previous_queries
