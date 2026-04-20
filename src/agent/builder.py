import ast
import json
from typing import Dict, List, Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from loguru import logger

from src.agent.prompts import failure_no_results_message
from src.agent.state import AgentState
from src.databases.qdrant.qdrant import QdrantDatabase
from src.databases.qdrant.search_embeddings import get_candidates, fetch_similar_qa_pairs
from src.tools.graph_context import enrich_candidates
from src.tools.ner import get_ner_result
from src.tools.tools import generate_sparql, validate_results
from src.utils.extract_previous_queries import extract_previous_queries
from src.utils.extract_qids import extract_all_qids
from src.utils.format_candidates_clean import format_candidates_clean
from src.wikidata.api import get_wikidata_labels


class SparqlAgentBuilder:
    """
    Builds and compiles the Text-to-SPARQL LangGraph Agent.
    Uses Dependency Injection for the LLM and Vector Database services.
    """

    def __init__(self, llm_service: Any, qdrant_service: QdrantDatabase):
        self.llm_service = llm_service
        self.qdrant_service = qdrant_service

        self.tools = self._initialize_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm_service.bind_tools(self.tools)

        logger.info("SparqlAgentBuilder initialized with LLM and Qdrant services.")

    def _initialize_tools(self) -> List[Any]:
        """Returns the list of tools available to the agent."""
        return [generate_sparql, validate_results]

    async def retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """Fetches NER keywords, Candidates from Qdrant, and Few-Shot Examples."""
        question = state.original_question
        attempts = state.attempts
        force_refresh = (attempts >= 2)

        ner_keywords = state.ner_keywords
        current_lang = state.language

        # 1. Extract NER Keywords
        if not ner_keywords or force_refresh:
            logger.debug(f"Extracting NER keywords for question: '{question}' (Force Refresh: {force_refresh})")
            try:
                ner_result = await get_ner_result(question)
                ner_keywords = [k.model_dump() for k in ner_result.keywords]
                current_lang = ner_result.lang
            except Exception as e:
                logger.exception("NER Extraction Failed")

        # 2. Retrieve Candidates (Semantic Search via Qdrant)
        # Note: Pass self.qdrant_service here if you refactor get_candidates to accept it
        candidates_map = await get_candidates(
            ner_keywords,
            lang=current_lang,
            qdrant_db=self.qdrant_service
        )
        # 3. Graph Context Enrichment
        await enrich_candidates(candidates_map)
        candidates_str = format_candidates_clean(candidates_map)

        # 4. Fetch Few-Shot Examples
        examples = await fetch_similar_qa_pairs(
            question,
            current_lang,
            qdrant_db=self.qdrant_service  # <--- ADD THIS
        )
        logger.info(f"Retrieval complete. Found {len(candidates_map)} candidate groups.")

        return {
            "ner_keywords": ner_keywords,
            "language": current_lang,
            "candidates": candidates_str,
            "examples": examples
        }

    async def llm_node(self, state: AgentState) -> Dict[str, List[BaseMessage]]:
        """Invokes the LLM to decide on the next action."""
        content = """You are an expert at converting user questions into SPARQL queries. 
        Your primary task is to use the provided tools to answer the user's question.
        **IMPORTANT**: Do not translate the question; keep the original language."""

        messages = [SystemMessage(content=content)] + state.messages

        logger.debug(f"Invoking LLM node (Attempt {state.attempts})")

        try:
            response = await self.llm_with_tools.ainvoke(messages)
            return {"messages": [response]}
        except Exception as e:
            logger.exception("LLM call failed in llm_node")
            return {"messages": [AIMessage(content=f"LLM call failed: {e}")]}

    async def tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Performs the tool call and handles potential errors."""
        original_question = state.original_question
        current_lang = state.language
        last_message = state.messages[-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            logger.warning("Tool node called, but no tool_calls found in the last message.")
            return {"attempts": state.attempts + 1}

        result_messages = []
        log_entries = []
        previous_queries = await extract_previous_queries(state)

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool = self.tools_by_name.get(tool_name)

            if not tool:
                logger.error(f"LLM tried to call unknown tool: {tool_name}")
                continue

            args = tool_call["args"]
            args["original_question"] = original_question
            args["language"] = current_lang
            args["candidates"] = state.candidates
            args["examples"] = state.examples
            args["ner_keywords"] = state.ner_keywords

            logger.info(f"Executing tool '{tool_name}'")

            try:
                observation = await tool.ainvoke(args)

                if isinstance(observation, dict) and 'log_data' in observation:
                    log_entries.append(observation.get("log_data", {}))

                # Handle Failed/Empty Queries explicitly
                if isinstance(observation, dict) and 'results' in observation and not observation['results']:
                    previous_queries.append(observation["sparql"])
                    content = failure_no_results_message.format(
                        user_task=original_question,
                        failed_query=observation["sparql"],
                        previous_queries="\n".join(previous_queries)
                    )
                    result_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
                    logger.warning("Tool returned no results. Triggering retry loop.")
                else:
                    result_messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
                    logger.debug("Tool executed successfully with results.")

            except Exception as e:
                logger.exception(f"Tool call '{tool_name}' failed")
                failure_message = f"Tool call failed with {type(e).__name__}: {str(e)}. Please try again with a different query."
                result_messages.append(ToolMessage(content=failure_message, tool_call_id=tool_call["id"]))

        return {
            "messages": result_messages,
            "attempts": state.attempts + 1,
            "log_data": log_entries,
            "current_results": observation.get("results") if isinstance(observation, dict) else None
        }

    async def validation_node(self, state: AgentState) -> Dict[str, List[SystemMessage]]:
        """Validates the SPARQL query results using LLM-as-a-judge."""
        original_question = state.original_question

        # Read safely from state, no parsing required!
        data_to_validate = state.current_results or state.messages[-1].content

        ids_to_lookup = list(extract_all_qids(data_to_validate))
        last_message = state.messages[-1]
        original_question = state.original_question
        raw_content = last_message.content

        logger.info("Running validation node on SPARQL results.")

        try:
            parsed_content = ast.literal_eval(raw_content)
            data_to_validate = parsed_content["results"] if isinstance(parsed_content,
                                                                       dict) and "results" in parsed_content else raw_content
        except (ValueError, SyntaxError):
            data_to_validate = raw_content

        ids_to_lookup = list(extract_all_qids(data_to_validate))
        enriched_context = ""

        # Fetch label definitions so the LLM Judge understands the QIDs
        if ids_to_lookup:
            try:
                labels_map = get_wikidata_labels(ids_to_lookup)
                if labels_map:
                    enriched_context = "\n\n**Entity Definitions (Context for Validator):**\n"
                    for qid, info_data in labels_map.items():
                        if not info_data: continue
                        first_item = info_data[0] if isinstance(info_data, list) else info_data
                        label = first_item.get('label', 'Unknown') if isinstance(first_item, dict) else str(first_item)
                        enriched_context += f"- {qid}: '{label}'\n"
                else:
                    enriched_context = "\n(No labels found for these IDs in Wikidata)"
            except Exception as e:
                logger.warning(f"Validation Error: API lookup failed for labels: {e}")
                enriched_context = "\n(Context lookup failed for specific IDs)"

        results_input = json.dumps(data_to_validate, indent=2, default=str) if not isinstance(data_to_validate,
                                                                                              str) else data_to_validate
        final_validation_input = f"Results:\n{results_input}\n{enriched_context}"

        validation_output = await validate_results.ainvoke({
            "question": original_question,
            "results": final_validation_input
        })

        # Process Validation Response
        if isinstance(validation_output, bool):
            is_valid = validation_output
            feedback_text = str(validation_output)
        else:
            val_str = str(validation_output).lower()
            is_valid = getattr(validation_output, 'is_valid', ("true" in val_str or "yes" in val_str))
            feedback_text = "Valid" if is_valid else str(validation_output)

        if is_valid:
            logger.success("Results validated successfully.")
            return {
                "messages": [SystemMessage(
                    content="The SPARQL results have been validated and appear correct. Please formulate your final answer.")]
            }
        else:
            logger.warning("Validation rejected the results. Sending feedback back to LLM.")
            content = (
                f"The previous query returned results, but the automated validator determined "
                f"they do not correctly answer the question: '{original_question}'.\n"
                f"Validator Feedback: {feedback_text}\n"
                f"Please analyze the previous query and results, and generate a CORRECTED SPARQL query."
            )
            return {"messages": [SystemMessage(content=content)]}

    def should_continue(self, state: AgentState) -> str:
        """Determines the next edge path in the graph."""
        messages = state.messages
        last_message = messages[-1]

        # Break infinite loops
        if state.attempts >= 5:
            logger.warning("Max attempts reached. Forcing END.")
            return END

        # If LLM finishes thinking
        if isinstance(last_message, AIMessage):
            if last_message.tool_calls:
                return "continue"
            return END

        # If Tool finishes executing
        if isinstance(last_message, ToolMessage):
            if len(messages) >= 2 and isinstance(messages[-2], AIMessage):
                last_ai_message = messages[-2]

                # Check if the tool executed was `generate_sparql`
                executed_tool_name = next(
                    (tc["name"] for tc in last_ai_message.tool_calls if tc["id"] == last_message.tool_call_id),
                    None
                )

                if executed_tool_name == generate_sparql.name:
                    content = last_message.content
                    if "Tool call failed" in content or "returned no results" in content:
                        return "continue"  # Go back to LLM to try again
                    return "validate"  # We got results, go validate them

            return "continue"

        return "continue"

    def compile(self) -> CompiledStateGraph:
        """Compiles and returns the executable LangGraph."""
        logger.info("Compiling the Sparql Agent Graph...")

        workflow = StateGraph(AgentState)

        # 1. Add Nodes
        workflow.add_node("retriever", self.retrieval_node)
        workflow.add_node("llm", self.llm_node)
        workflow.add_node("tool_executor", self.tool_node)
        workflow.add_node("validator", self.validation_node)

        # 2. Define Entry Point
        workflow.set_entry_point("retriever")

        # 3. Define Standard Edges
        workflow.add_edge("retriever", "llm")

        # 4. Define Conditional Edges
        workflow.add_conditional_edges(
            "llm",
            self.should_continue,
            {
                "continue": "tool_executor",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "tool_executor",
            self.should_continue,
            {
                "validate": "validator",
                "continue": "retriever",  # If it failed, retrieve fresh context and try again
                END: END
            }
        )

        workflow.add_conditional_edges(
            "validator",
            self.should_continue,
            {
                "continue": "retriever",  # Loop back through retriever and LLM if validation failed
                END: END,
            },
        )

        return workflow.compile()
