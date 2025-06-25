# import json
# import os
# from typing import Any, Optional, Dict, List
#
# from langchain_ollama import OllamaLLM, ChatOllama
# from pydantic import BaseModel
# from qdrant_client.http.models import ScoredPoint
# from rdflib.plugins.sparql import prepareQuery
#
# from src.agent.state import GraphState
# from src.databases.qdrant.search_embeddings import fetch_similar_entities, search_embeddings
# from src.llm.generic_chat import generic_chat
# from src.templates.ner import extract_entities
# from src.templates.sparql import sparql_template
# from src.utils.json__extraction import get_json_response
# from src.wikidata.api import execute_sparql_query
#
#
# llm = ChatOllama(
#     model=os.getenv("OLLAMA_MODEL", "llama3.3:70b"),
#     base_url="https://llama3.finki.ukim.mk",
#     temperature=0,
# )
#
#
# def agent_router_node(state: GraphState) -> Dict[str, Any]:
#     print("---ROUTING---")
#     if state.get("error"):
#         print(f"Error detected: {state['error']}. Halting.")
#         return {"next_action": "error"}
#
#     prompt_parts = [f"The user's question is: '{state['question']}'."]
#
#     if state.get("error"):
#         prompt_parts.append(f"An error occurred in the last step: {state['error']}. Please decide how to proceed.")
#     if not state.get("labels"):
#         prompt_parts.append("I need to extract entities from the question.")
#         action = "Extract_Labels"
#     elif not state.get("similar_entities"):
#         prompt_parts.append("I have the labels, now I need to find similar entities in Wikidata.")
#         action = "Find_Similar_Entities"
#     elif not state.get("examples"):
#         prompt_parts.append("I also need to find similar examples to help with query generation.")
#         action = "Find_Similar_Examples"
#     elif not state.get("sparql_query"):
#         prompt_parts.append("I have entities and examples. I am ready to generate the SPARQL query.")
#         action = "Generate_SPARQL"
#     elif not state.get("results"):
#         prompt_parts.append("I have a SPARQL query. I need to execute it.")
#         action = "Execute_SPARQL"
#     else:
#         prompt_parts.append("I have executed the query and have the results. I should now formulate the final answer.")
#         action = "finish"
#
#     print(f"Next action: {action}")
#     return {"next_action": action}
#
#
# def extract_labels_node(state: GraphState) -> Dict[str, Any]:
#     print("---EXTRACTING LABELS---")
#     question = state["question"]
#     result = tool_get_ner_results({"question": question})
#     if result.success:
#         return {"labels": result.payload["labels"], "lang": result.payload["lang"], "error": None}
#     else:
#         return {"error": result.error}
#
#
# def find_entities_node(state: GraphState) -> Dict[str, Any]:
#     print("---FINDING SIMILAR ENTITIES---")
#     result = tool_fetch_similar_entities({"labels": state["labels"], "lang": state["lang"]})
#     if result:
#         return {"similar_entities": result.payload}
#     else:
#         return {"error": result.error}
#
#
# def find_examples_node(state: GraphState) -> Dict[str, Any]:
#     print("---FINDING EXAMPLES---")
#     result = tool_get_examples({"question": state["question"]})
#     if result.success:
#         # FIX: The payload is already a list of dictionaries. No need to call model_dump().
#         return {"examples": result.payload}
#     else:
#         return {"error": result.error}
#
#
# def generate_sparql_node(state: GraphState) -> Dict[str, Any]:
#     """Node to call the Generate_SPARQL tool."""
#     print("---GENERATING SPARQL---")
#     result = tool_generate_sparql({
#         "question": state["question"],
#         "examples": state["examples"],
#         "similar_entities": state["similar_entities"],
#     })
#     if result.success:
#         return {"sparql_query": result.payload["sparql"], "error": None}
#     else:
#         return {"error": result.error}
#
#
# def execute_sparql_node(state: GraphState) -> Dict[str, Any]:
#     """Node to call the Execute_SPARQL tool."""
#     print("---EXECUTING SPARQL---")
#     result = tool_execute_sparql({"sparql": state["sparql_query"]})
#     if result.success:
#         return {"results": result.payload["results"], "error": None}
#     else:
#         return {"error": result.error}
#
#
# def format_answer_node(state: GraphState) -> Dict[str, Any]:
#     """Node to format the final answer for the user."""
#     print("---FORMATTING FINAL ANSWER---")
#     # This could be a simple summary or a sophisticated LLM call
#     # to make the JSON results human-readable.
#     final_answer = json.dumps(state["results"], indent=2)
#     return {"final_answer": f"Query executed successfully. Results:\n{final_answer}"}
# class ToolResult(BaseModel):
#     success: bool
#     payload: Optional[Any]
#     error: Optional[str]
#
#
# def safe_tool(fn):
#     def wrapper(inputs: Any) -> ToolResult:
#         if isinstance(inputs, str):
#             try:
#                 inputs = json.loads(inputs)
#             except json.JSONDecodeError:
#                 pass
#         try:
#             result = fn(inputs)
#             return ToolResult(success=True, payload=result, error=None)
#         except Exception as e:
#             return ToolResult(success=False, error=f"{fn.__name__} failed: {e}", payload=None)
#
#     return wrapper
#
#
# @safe_tool
# def tool_get_ner_results(inputs: Dict[str, Any]) -> Any:
#     return extract_entities(inputs["question"])
#
#
# @safe_tool
# def tool_fetch_similar_entities(inputs: Dict[str, Any]) -> List[Any]:
#     labels = inputs["labels"]
#     lang = inputs.get("lang", "en")
#     return fetch_similar_entities(labels, lang)
#
#
# # @safe_tool
# # def get_schema(entities: List[Any]):
# #     schema = get_entities_schema(entities)
# #     return format_schema_output(schema, entities)
#
#
# @safe_tool
# def tool_get_examples(inputs: Dict[str, Any]) -> List[ScoredPoint]:
#     question = inputs["question"]
#     return search_embeddings(question, collection_name="lcquad2_0")
#
#
# def validate_sparql(query: str):
#     try:
#         prefixes = """
#             PREFIX wd: <http://www.wikidata.org/entity/>
#             PREFIX wdt: <http://www.wikidata.org/prop/direct/>
#         """
#
#         full_query = prefixes + query
#         prepareQuery(full_query)
#         return True, "Query is valid."
#     except Exception as e:
#         return False, f"Validation error: {e}"
#
#
# @safe_tool
# def tool_generate_sparql(inputs: Dict[str, Any]) -> Dict[str, Any]:
#     if isinstance(inputs, str):
#         inputs = json.loads(inputs)
#     question = inputs["question"]
#     examples = inputs["examples"]
#     sim_entities = inputs["similar_entities"]
#
#     if not sim_entities:
#         raise ValueError("Cannot generate SPARQL query without any similar entities.")
#
#     sparql_prompt = sparql_template(question, examples, sim_entities)
#     sparql = get_json_response(
#         sparql_prompt,
#         list_name="sparql",
#         system_message="You are a SPARQL query generator."
#     )
#     if not sparql:
#         raise ValueError("Missing 'sparql' in generation response")
#
#     is_valid, msg = validate_sparql(sparql["sparql"])
#     if not is_valid:
#         raise ValueError(f"Invalid SPARQL: {sparql} {msg}")
#
#     return {"sparql": sparql}
#
#
# @safe_tool
# def tool_execute_sparql(inputs: Dict[str, Any]) -> Dict[str, Any]:
#     query = inputs["sparql"]
#     results = execute_sparql_query(query)
#     return {"query": query, "results": results}
#
#
# @safe_tool
# def tool_error_handling(inputs: Dict[str, Any]) -> Dict[str, Any]:
#     error_msg = inputs.get("error", "Unknown error")
#     llm_response = generic_chat(
#         message=f"Tool error occurred. Please analyze and suggest next steps. Error: {error_msg}"
#     )
#     return {"message": llm_response}
#
# tools = [tool_get_ner_results, tool_fetch_similar_entities, tool_get_examples, tool_error_handling, tool_execute_sparql, tool_generate_sparql]
# tools_by_name = {tool.name: tool for tool in tools}
# llm_with_tools = llm.bind_tools(tools)
