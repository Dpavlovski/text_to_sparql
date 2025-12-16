import ast
import json

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from src.agent.prompts import failure_no_results_message
from src.agent.state import AgentState
from src.databases.qdrant.search_embeddings import get_candidates, fetch_similar_qa_pairs
from src.llm.llm_provider import llm_provider
from src.tools.graph_context import enrich_candidates
from src.tools.ner import get_ner_result
from src.tools.tools import generate_sparql, validate_results
from src.utils.extract_previous_queries import extract_previous_queries
from src.utils.extract_qids import extract_all_qids
from src.utils.format_candidates_clean import format_candidates_clean
from src.wikidata.api import get_wikidata_labels

tools = [generate_sparql, validate_results]
tools_by_name = {tool.name: tool for tool in tools}
llm = llm_provider.get_model("gpt-4.1-mini")
llm_with_tools = llm.bind_tools(tools)


async def llm_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    """Invokes the LLM to decide on the next action."""
    content = """ You are an expert at converting user questions into SPARQL queries. 
    Your primary task is to use the provided tools to answer the user's question.
    **IMPORTANT**: Do not translate the question keep the original language."""

    try:
        response = await llm_with_tools.ainvoke(
            [SystemMessage(content=content)] + state["messages"]
        )
        return {"messages": [response]}
    except Exception as e:
        print(f"ERROR: An exception occurred in the llm_node: {e}")
        return {"messages": [AIMessage(content=f"LLM call failed: {e}")]}


async def retrieval_node(state: AgentState):
    """Fetches NER keywords, Candidates, and Examples."""
    question = state["original_question"]
    attempts = state.get("attempts", 0)

    force_refresh = (attempts >= 2)

    ner_keywords = state.get("ner_keywords", [])
    current_lang = state.get("language", "en")

    if not ner_keywords or force_refresh:
        try:
            ner_result = await get_ner_result(question)
            ner_keywords = [k.model_dump() for k in ner_result.keywords]
            current_lang = ner_result.lang
        except Exception as e:
            print(f"NER Extraction Failed: {e}")

    candidates_map = await get_candidates(ner_keywords, lang=current_lang)

    await enrich_candidates(candidates_map)

    candidates_str = format_candidates_clean(candidates_map)

    examples = await fetch_similar_qa_pairs(question, current_lang)

    return {
        "ner_keywords": ner_keywords,
        "language": current_lang,
        "candidates": candidates_str,
        "examples": examples
    }


async def tool_node(state: AgentState):
    """Performs the tool call and handles potential errors."""
    original_question = state["original_question"]
    current_lang = state.get("language", "en")

    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"attempts": state['attempts'] + 1}

    result_messages = []
    log_entries = []

    previous_queries = await extract_previous_queries(state)

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        args = tool_call["args"]

        args["original_question"] = original_question
        args["language"] = current_lang
        args["candidates"] = state.get("candidates", "")
        args["examples"] = state.get("examples", "")
        args["ner_keywords"] = state.get("ner_keywords", [])

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


async def validation_node(state: AgentState):
    """Validates the SPARQL query results."""
    last_message = state["messages"][-1]
    original_question = state["original_question"]
    raw_content = last_message.content

    try:
        parsed_content = ast.literal_eval(raw_content)
        if isinstance(parsed_content, dict) and "results" in parsed_content:
            data_to_validate = parsed_content["results"]
        else:
            data_to_validate = raw_content
    except (ValueError, SyntaxError):
        data_to_validate = raw_content

    ids_to_lookup = list(extract_all_qids(data_to_validate))
    enriched_context = ""

    if ids_to_lookup:
        try:
            labels_map = get_wikidata_labels(ids_to_lookup)
            if labels_map:
                enriched_context = "\n\n**Entity Definitions (Context for Validator):**\n"
                for qid, info_data in labels_map.items():
                    if not info_data: continue
                    first_item = info_data[0] if isinstance(info_data, list) else info_data
                    if isinstance(first_item, dict):
                        label = first_item.get('label', 'Unknown')
                        desc = first_item.get('description', 'No description')
                        enriched_context += f"- {qid}: '{label}' ({desc})\n"
                    elif isinstance(first_item, str):
                        enriched_context += f"- {qid}: '{first_item}'\n"
                    else:
                        enriched_context += f"- {qid}: {str(first_item)}\n"
            else:
                enriched_context = "\n(No labels found for these IDs in Wikidata)"
        except Exception as e:
            print(f"DEBUG Validation Error: API lookup failed: {e}")
            enriched_context = f"\n(Context lookup failed for specific IDs)"

    if not isinstance(data_to_validate, str):
        results_input = json.dumps(data_to_validate, indent=2, default=str)
    else:
        results_input = data_to_validate

    final_validation_input = f"Results:\n{results_input}\n{enriched_context}"

    validation_output = await validate_results.ainvoke(
        {"question": original_question, "results": final_validation_input}
    )

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

    if state["attempts"] >= 5:
        return END

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            return "continue"
        else:
            return END

    if isinstance(last_message, ToolMessage):
        if len(messages) >= 2 and isinstance(messages[-2], AIMessage):
            last_ai_message = messages[-2]
            corresponding_tool_name = None
            for tc in last_ai_message.tool_calls:
                if tc["id"] == last_message.tool_call_id:
                    corresponding_tool_name = tc["name"]
                    break

            if corresponding_tool_name == generate_sparql.name:
                content = last_message.content
                if "Tool call failed" in content or "returned no results" in content:
                    return "continue"
                return "validate"

        return "continue"

    return "continue"


def create_sparql_agent() -> CompiledStateGraph:
    """Builds and compiles the LangGraph agent."""
    agent_builder = StateGraph(AgentState)

    agent_builder.add_node("retriever", retrieval_node)
    agent_builder.add_node("llm", llm_node)
    agent_builder.add_node("tool_executor", tool_node)
    agent_builder.add_node("validator", validation_node)

    agent_builder.set_entry_point("retriever")

    agent_builder.add_edge("retriever", "llm")

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
            "continue": "retriever",
            END: END
        }
    )

    agent_builder.add_conditional_edges(
        "validator",
        should_continue,
        {
            "continue": "retriever",
            END: END,
        },
    )

    return agent_builder.compile()
