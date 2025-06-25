from typing import Any, Dict

from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph

from src.databases.qdrant.search_embeddings import fetch_similar_entities, search_embeddings
from src.llm.generic_llm import generic_llm
from src.templates.ner import extract_entities, NERResponse
from src.templates.sparql import get_sparql_query
from src.utils.format_examples import format_examples


@tool("ner_tool")
def ner(question: str) -> NERResponse:
    """Extract keywords from a question.

    Args:
        question: question string
    """
    return extract_entities(question)


@tool("fins_similar_entities_tool", args_schema=NERResponse)
def find_similar_entities(ner_response: NERResponse) -> Dict[str, Any]:
    """Finds and separates similar items and properties based on extracted keywords.

        This tool is designed to process the output of the 'ner' tool.
        It takes the 'keywords' list and the 'lang' string
        produced by the 'ner' tool as its arguments.

        Args:
            ner_response: The JSON format of keywords that was extracted by the 'ner' tool.        """
    keywords = ner_response[0]
    lang = ner_response[1]
    return fetch_similar_entities(keywords, lang)


@tool
def find_examples(question: str) -> str:
    """Finds similar question-query examples to the user's question to be used as context.

    Args:
        question: question string
    """
    return format_examples(search_embeddings(question, collection_name="lcquad2_0"))


@tool
def generate_sparql(question: str, examples: str, sim_entities: Dict[str, Any]) -> dict[str, Any]:
    """Generate the final SPARQL query based on the question, retrieved examples, and similar entities. This is the final step.

    Args:
        question: The original user question.
        examples: The list of examples found by the 'find_examples' tool.
        sim_entities: The list of entities found by the 'find_similar_entities' tool.
    """
    return get_sparql_query(question, examples, sim_entities)


tools = [ner, find_similar_entities, find_examples, generate_sparql]
tools_by_name = {tool.name: tool for tool in tools}
llm = generic_llm()
llm_with_tools = llm.bind_tools(tools)


# Nodes
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are an expert at converting user questions into SPARQL queries to answer them."
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState):
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

messages = [SystemMessage("You need to answer the question by generating SPARQL queries for Wikidata."),
            HumanMessage(content="What is the boiling point of water?")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
