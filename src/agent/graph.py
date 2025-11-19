import re
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from src.agent.prompts import failure_no_results_message
from src.agent.state import AgentState
from src.llm.llm_provider import llm_provider
from src.tools.tools import generate_sparql, validate_results

tools = [generate_sparql, validate_results]
tools_by_name = {tool.name: tool for tool in tools}
llm = llm_provider.get_model("gpt-4.1-mini")
llm_with_tools = llm.bind_tools(tools)


def llm_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    """Invokes the LLM to decide on the next action or to generate a final response."""
    try:
        response = llm_with_tools.invoke(
            [
                SystemMessage(
                    content="You are an expert at converting user questions into SPARQL queries. "
                            "Your primary task is to use the provided tools to answer the user's question. "
                            "**IMPORTANT**: Always use the user's original, untranslated question in the 'question' argument for the tool. Do not modify or translate it."
                )

            ]
            + state["messages"]
        )
        return {"messages": [response]}

    except Exception as e:
        print("=" * 80)
        print(f"ERROR: An exception occurred in the llm_node, likely during the call to the llm.")
        print(f"       Error details: {e}")
        print("=" * 80)
        error_message = f"LLM call failed: {e}"
        return {"messages": [ToolMessage(content=error_message, tool_call_id="llm_error")]}


async def tool_node(state: AgentState):
    """Performs the tool call and handles potential errors."""
    original_question = state["original_question"]
    last_message = state["messages"][-1]
    result_messages = []
    log_entries = []

    previous_queries = await extract_previous_queries(state)

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        args = tool_call["args"]
        args["original_question"] = original_question

        try:
            observation = await tool.ainvoke(args)
            if isinstance(observation, dict) and 'log_data' in observation:
                log_entries.append(observation.get("log_data", {}))

            if isinstance(observation, dict) and 'results' in observation and (
                    observation['results'] is None or observation['results'] == []):
                previous_queries.append(observation["sparql"])
                content = failure_no_results_message.format(
                    user_task=original_question,
                    failed_query=observation["sparql"],
                    previous_queries="\n".join(previous_queries)
                )
                result_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
            else:
                result_messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            failure_message = f"Tool call failed with {error_type}: {error_message}. Please try again with a different query."
            result_messages.append(ToolMessage(content=failure_message, tool_call_id=tool_call["id"]))

    return {
        "messages": result_messages,
        "attempts": state['attempts'] + 1,
        "log_data": log_entries
    }


async def extract_previous_queries(state: AgentState) -> list[Any]:
    # Regex to find SPARQL queries in text
    sparql_pattern = re.compile(
        r"""\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b # SPARQL keywords
        .*?                                     # Lazy match of the full query
        (?<=})                                  # Ensure it ends with a closing brace
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )

    # Collect previously attempted queries from ToolMessages
    previous_queries = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            match = sparql_pattern.search(msg.content)
            if match:
                sparql_query = match.group(0).strip()
                previous_queries.append(sparql_query)
    return previous_queries


# --- Conditional Edge ---

async def validation_node(state: AgentState):
    """
    Validates the SPARQL query results.
    """
    last_message = state["messages"][-1]
    original_question = state["original_question"]

    # The result is in the content of the last message
    results = last_message.content

    is_valid = await validate_results.ainvoke(
        {"question": original_question, "results": results}
    )

    if is_valid:
        return {"messages": [ToolMessage(content="Results are valid.", tool_call_id="validation")]}
    else:
        # If the results are not valid, provide feedback to the LLM
        content = "The previous query returned results that did not answer the question. Please try again."
        return {"messages": [ToolMessage(content=content, tool_call_id="validation")]}


def should_continue(state: AgentState) -> str:
    """Determines whether to continue the loop or end."""
    last_message = state["messages"][-1]

    if state["attempts"] >= 5:
        print("Maximum attempts reached. Ending the process.")
        return END

    # The AIMessage will have tool_calls if the LLM is requesting a tool.
    # Otherwise, it's a response to the user and we can end.
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            return "continue"
        else:
            return END

    # If the last message is a ToolMessage, we always want to continue
    # so the LLM can process the tool's output.
    return "continue"


def create_sparql_agent() -> CompiledStateGraph:
    """Builds and compiles the LangGraph agent for SPARQL generation."""
    agent_builder = StateGraph(AgentState)

    agent_builder.add_node("llm", llm_node)
    agent_builder.add_node("tool_executor", tool_node)
    agent_builder.add_node("validator", validation_node)

    agent_builder.set_entry_point("llm")

    agent_builder.add_conditional_edges(
        "llm",
        should_continue,
        {
            "continue": "tool_executor",
            END: END,
        },
    )

    agent_builder.add_conditional_edges(
        "tool_executor",
        should_continue,
        {
            "validate": "validator",
            "continue": "llm",  # If there's an error or no results, go back to the LLM
            END: END
        }
    )

    agent_builder.add_conditional_edges(
        "validator",
        should_continue,
        {
            "continue": "llm",  # If validation fails, go back to the LLM
            END: END,
        },
    )

    return agent_builder.compile()
