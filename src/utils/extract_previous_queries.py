import re
from typing import Any

from langchain_core.messages import ToolMessage

from src.agent.state import AgentState


async def extract_previous_queries(state: AgentState) -> list[Any]:
    sparql_pattern = re.compile(
        r"""\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b
        .*?
        (?<=})
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )

    previous_queries = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            match = sparql_pattern.search(msg.content)
            if match:
                sparql_query = match.group(0).strip()
                previous_queries.append(sparql_query)
    return previous_queries
