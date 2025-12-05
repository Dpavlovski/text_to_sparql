import ast
import json
import re
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from src.agent.prompts import failure_no_results_message
from src.agent.state import AgentState
from src.llm.llm_provider import llm_provider
from src.tools.tools import generate_sparql, validate_results
from src.utils.format_uri import extract_id_from_uri
from src.wikidata.api import get_wikidata_labels

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
    # Retrieve language from state, default to 'en' for backward compatibility
    current_lang = state.get("language", "en")

    last_message = state["messages"][-1]
    result_messages = []
    log_entries = []

    previous_queries = await extract_previous_queries(state)

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        args = tool_call["args"]

        # --- INJECTION START ---
        # We manually inject state variables into the tool arguments
        args["original_question"] = original_question
        args["language"] = current_lang
        # --- INJECTION END ---

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
    sparql_pattern = re.compile(
        r"""\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b # SPARQL keywords
        .*?                                     # Lazy match of the full query
        (?<=})                                  # Ensure it ends with a closing brace
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


def extract_all_qids(data):
    """Recursively finds all strings looking like Q-IDs in a nested JSON object."""
    qids = set()

    if isinstance(data, dict):
        for key, value in data.items():
            # If we see a "value" key (SPARQL format), check it specifically
            if key == "value" and isinstance(value, str):
                qid = extract_id_from_uri(value)
                if qid: qids.add(qid)
            # Otherwise recurse
            else:
                qids.update(extract_all_qids(value))

    elif isinstance(data, list):
        for item in data:
            qids.update(extract_all_qids(item))

    elif isinstance(data, str):
        # Direct string check
        qid = extract_id_from_uri(data)
        if qid: qids.add(qid)

    return qids


async def validation_node(state: AgentState):
    """
    Validates the SPARQL query results.
    """
    last_message = state["messages"][-1]
    original_question = state["original_question"]
    raw_content = last_message.content

    # 1. Parse Data
    try:
        parsed_content = ast.literal_eval(raw_content)
        if isinstance(parsed_content, dict) and "results" in parsed_content:
            data_to_validate = parsed_content["results"]
        else:
            data_to_validate = raw_content
    except (ValueError, SyntaxError):
        data_to_validate = raw_content

    # 2. Extract IDs using the robust helper
    ids_to_lookup = list(extract_all_qids(data_to_validate))

    # Debug print to see what we found (Check your console)
    print(f"DEBUG Validation: Found IDs to lookup: {ids_to_lookup}")

    # 3. Fetch Context
    enriched_context = ""
    if ids_to_lookup:
        try:
            # Fetch labels
            labels_map = get_wikidata_labels(ids_to_lookup)

            if labels_map:
                enriched_context = "\n\n**Entity Definitions (Context for Validator):**\n"
                for qid, info_data in labels_map.items():
                    # Skip empty data
                    if not info_data:
                        continue

                    # Normalize data: Ensure we have a single item to look at
                    first_item = info_data[0] if isinstance(info_data, list) else info_data

                    # CASE A: It is a Dictionary (Ideal case: Label + Desc)
                    if isinstance(first_item, dict):
                        label = first_item.get('label', 'Unknown')
                        desc = first_item.get('description', 'No description')
                        enriched_context += f"- {qid}: '{label}' ({desc})\n"

                    # CASE B: It is just a String (Label only)
                    elif isinstance(first_item, str):
                        enriched_context += f"- {qid}: '{first_item}'\n"

                    # CASE C: Fallback
                    else:
                        enriched_context += f"- {qid}: {str(first_item)}\n"
            else:
                enriched_context = "\n(No labels found for these IDs in Wikidata)"

        except Exception as e:
            # We log the error but allow validation to proceed without context
            print(f"DEBUG Validation Error: API lookup failed: {e}")
            enriched_context = f"\n(Context lookup failed for specific IDs)"

    # 4. Convert Results to String for LLM
    if not isinstance(data_to_validate, str):
        results_input = json.dumps(data_to_validate, indent=2, default=str)
    else:
        results_input = data_to_validate

    # 5. Append Context
    final_validation_input = f"Results:\n{results_input}\n{enriched_context}"

    # 6. Invoke Validator
    validation_output = await validate_results.ainvoke(
        {"question": original_question, "results": final_validation_input}
    )

    # ... [Rest of your boolean logic remains the same] ...
    if isinstance(validation_output, bool):
        is_valid = validation_output
        feedback_text = str(validation_output)
    else:
        val_str = str(validation_output).lower()
        if hasattr(validation_output, 'is_valid'):
            is_valid = validation_output.is_valid
            feedback_text = "Valid" if is_valid else "Invalid"
        else:
            is_valid = "true" in val_str or "yes" in val_str
            feedback_text = str(validation_output)

    if is_valid:
        return {
            "messages": [
                SystemMessage(
                    content="The SPARQL results have been validated and appear correct. Please formulate your final answer to the user based on these results."
                )
            ]
        }
    else:
        content = (
            f"The previous query returned results, but the automated validator determined "
            f"they do not correctly answer the question: '{original_question}'.\n"
            f"Validator Feedback: {feedback_text}\n"
            f"Please analyze the previous query and results, and generate a CORRECTED SPARQL query."
        )
        return {
            "messages": [SystemMessage(content=content)]
        }


def should_continue(state: AgentState) -> str:
    """Determines the path of the graph execution."""
    messages = state["messages"]
    last_message = messages[-1]

    # 1. Check Max Attempts
    if state["attempts"] >= 5:
        print("Maximum attempts reached. Ending the process.")
        return END

    # 2. Handle LLM Response (AIMessage)
    if isinstance(last_message, AIMessage):
        # If the LLM wants to run a tool, let it continue to 'tool_executor'
        if last_message.tool_calls:
            return "continue"
        # Otherwise, the LLM has finished answering
        else:
            return END

    # 3. Handle Tool Output (ToolMessage)
    # This is where your bug was. We need to decide: Go to Validator OR Go back to LLM?
    if isinstance(last_message, ToolMessage):

        # We need to find which tool produced this message.
        # We look at the preceding AIMessage (index -2) to match the tool_call_id.
        if len(messages) >= 2 and isinstance(messages[-2], AIMessage):
            last_ai_message = messages[-2]

            # Find the tool call that matches this ToolMessage's ID
            corresponding_tool_name = None
            for tc in last_ai_message.tool_calls:
                if tc["id"] == last_message.tool_call_id:
                    corresponding_tool_name = tc["name"]
                    break

            # LOGIC: If the tool was 'generate_sparql', we usually want to validate it.
            # However, if the tool returned an error (e.g., "Tool call failed"),
            # skip validation and go back to LLM to fix it.
            if corresponding_tool_name == generate_sparql.name:

                # Simple heuristic to detect error messages generated in tool_node
                content = last_message.content
                if "Tool call failed" in content or "returned no results" in content:
                    return "continue"  # Go to LLM to fix the error

                # If it looks like a successful query result, go to Validator
                return "validate"

        # For all other tools (or if logic fails), go back to LLM to interpret results
        return "continue"

    # 4. Handle Validator Output (SystemMessage)
    # If the last message was the system message from validation_node,
    # we return "continue" so the edge maps to "llm" (see add_conditional_edges)
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
