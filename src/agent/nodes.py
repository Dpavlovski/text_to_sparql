# import ast
# import asyncio
# import json
#
# from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
#
# from src.agent.graph import llm_with_tools, tools_by_name
# from src.agent.prompts import failure_no_results_message
# from src.agent.state import AgentState
# from src.databases.qdrant.search_embeddings import get_candidates, fetch_similar_qa_pairs
# from src.tools.ner import extract_entities, Keyword
# from src.tools.schema import get_entity_schema
# from src.tools.tools import validate_results
# from src.utils.extract_previous_queries import extract_previous_queries
# from src.utils.extract_qids import extract_all_qids
# from src.wikidata.api import get_wikidata_labels
#
#
# async def llm_node(state: AgentState) -> dict[str, list[BaseMessage]]:
#     """Invokes the LLM to decide on the next action."""
#     content = "You are an expert at converting user questions into SPARQL queries. "
#     "Your primary task is to use the provided tools to answer the user's question. "
#     "**IMPORTANT**: Do not translate the question keep the original language."
#
#     try:
#         response = llm_with_tools.ainvoke(
#             [SystemMessage(content=content)] + state["messages"]
#         )
#         return {"messages": [response]}
#     except Exception as e:
#         print(f"ERROR: An exception occurred in the llm_node.")
#         error_message = f"LLM call failed: {e}"
#         return {"messages": [ToolMessage(content=error_message, tool_call_id="llm_error")]}
#
#
# async def retrieval_node(state: AgentState):
#     """Fetches NER keywords, Candidates, Schema, and Examples."""
#     question = state["original_question"]
#     attempts = state.get("attempts", 0)
#
#     force_refresh = (attempts >= 2)
#
#     # 1. Get Keywords
#     keywords_raw = state.get("ner_keywords", [])
#     current_lang = state.get("language", "en")
#
#     if not keywords_raw or force_refresh:
#         print(f"--- Running NER Extraction (Attempt {attempts}) ---")
#         if force_refresh:
#             print("!!! Force Refreshing NER: Previous candidates might be wrong !!!")
#
#         try:
#             ner_result = await extract_entities(question)
#             keywords_raw = [k.model_dump() for k in ner_result.keywords]
#             current_lang = ner_result.lang
#         except Exception as e:
#             print(f"NER Extraction Failed: {e}")
#             keywords_raw = []
#     else:
#         print("Using cached NER keywords.")
#
#     # 2. Convert to Objects for Candidate Search
#     keyword_objs = [Keyword(**k) for k in keywords_raw]
#
#     # 3. Get Candidates
#     candidates_map = await get_candidates(keyword_objs, lang=current_lang)
#
#     # 4. Get Examples
#     examples = await fetch_similar_qa_pairs(question, current_lang)
#
#     # 5. Construct Schema Context
#     schema_tasks = []
#     candidate_meta = []
#
#     for keyword_val, candidate_list in candidates_map.items():
#         for candidate in candidate_list[:2]:
#             qid = candidate.get('id')
#             label = candidate.get('label', 'Unknown')
#             desc = candidate.get('description', '')
#
#             if qid:
#                 schema_tasks.append(get_entity_schema(qid))
#                 candidate_meta.append((keyword_val, label, desc, qid))
#
#     if schema_tasks:
#         schema_results = await asyncio.gather(*schema_tasks)
#     else:
#         schema_results = []
#
#     schema_context = ""
#     for i, (kw, label, desc, qid) in enumerate(candidate_meta):
#         result_text = schema_results[i]
#         if result_text:
#             schema_context += f"### Mention: '{kw}' -> Entity: '{label}' ({qid})\n"
#             schema_context += f"Description: {desc}\n"
#             schema_context += f"Schema Usage:\n{result_text}\n\n"
#
#     return {
#         "ner_keywords": keywords_raw,
#         "language": current_lang,
#         "candidates": str(candidates_map),
#         "schema_context": schema_context,
#         "examples": examples
#     }
#
#
# async def tool_node(state: AgentState):
#     """Performs the tool call and handles potential errors."""
#     original_question = state["original_question"]
#     current_lang = state.get("language", "en")
#
#     last_message = state["messages"][-1]
#     result_messages = []
#     log_entries = []
#
#     previous_queries = await extract_previous_queries(state)
#
#     for tool_call in last_message.tool_calls:
#         tool = tools_by_name[tool_call["name"]]
#         args = tool_call["args"]
#
#         args["original_question"] = original_question
#         args["language"] = current_lang
#         args["schema_context"] = state.get("schema_context", "")
#         args["candidates"] = state.get("candidates", "")
#         args["examples"] = state.get("examples", "")
#
#         try:
#             observation = await tool.ainvoke(args)
#             if isinstance(observation, dict) and 'log_data' in observation:
#                 log_entries.append(observation.get("log_data", {}))
#
#             if isinstance(observation, dict) and 'results' in observation and (
#                     observation['results'] is None or observation['results'] == []):
#                 previous_queries.append(observation["sparql"])
#                 content = failure_no_results_message.format(
#                     user_task=original_question,
#                     failed_query=observation["sparql"],
#                     previous_queries="\n".join(previous_queries)
#                 )
#                 result_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
#             else:
#                 result_messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
#
#         except Exception as e:
#             error_type = type(e).__name__
#             error_message = str(e)
#             failure_message = f"Tool call failed with {error_type}: {error_message}. Please try again with a different query."
#             result_messages.append(ToolMessage(content=failure_message, tool_call_id=tool_call["id"]))
#
#     return {
#         "messages": result_messages,
#         "attempts": state['attempts'] + 1,
#         "log_data": log_entries
#     }
#
#
# async def validation_node(state: AgentState):
#     """Validates the SPARQL query results."""
#     last_message = state["messages"][-1]
#     original_question = state["original_question"]
#     raw_content = last_message.content
#
#     try:
#         parsed_content = ast.literal_eval(raw_content)
#         if isinstance(parsed_content, dict) and "results" in parsed_content:
#             data_to_validate = parsed_content["results"]
#         else:
#             data_to_validate = raw_content
#     except (ValueError, SyntaxError):
#         data_to_validate = raw_content
#
#     ids_to_lookup = list(extract_all_qids(data_to_validate))
#     enriched_context = ""
#
#     if ids_to_lookup:
#         try:
#             labels_map = get_wikidata_labels(ids_to_lookup)
#             if labels_map:
#                 enriched_context = "\n\n**Entity Definitions (Context for Validator):**\n"
#                 for qid, info_data in labels_map.items():
#                     if not info_data: continue
#                     first_item = info_data[0] if isinstance(info_data, list) else info_data
#                     if isinstance(first_item, dict):
#                         label = first_item.get('label', 'Unknown')
#                         desc = first_item.get('description', 'No description')
#                         enriched_context += f"- {qid}: '{label}' ({desc})\n"
#                     elif isinstance(first_item, str):
#                         enriched_context += f"- {qid}: '{first_item}'\n"
#                     else:
#                         enriched_context += f"- {qid}: {str(first_item)}\n"
#             else:
#                 enriched_context = "\n(No labels found for these IDs in Wikidata)"
#         except Exception as e:
#             print(f"DEBUG Validation Error: API lookup failed: {e}")
#             enriched_context = f"\n(Context lookup failed for specific IDs)"
#
#     if not isinstance(data_to_validate, str):
#         results_input = json.dumps(data_to_validate, indent=2, default=str)
#     else:
#         results_input = data_to_validate
#
#     final_validation_input = f"Results:\n{results_input}\n{enriched_context}"
#
#     validation_output = await validate_results.ainvoke(
#         {"question": original_question, "results": final_validation_input}
#     )
#
#     if isinstance(validation_output, bool):
#         is_valid = validation_output
#         feedback_text = str(validation_output)
#     else:
#         val_str = str(validation_output).lower()
#         if hasattr(validation_output, 'is_valid'):
#             is_valid = validation_output.is_valid
#             feedback_text = "Valid" if is_valid else "Invalid"
#         else:
#             is_valid = "true" in val_str or "yes" in val_str
#             feedback_text = str(validation_output)
#
#     if is_valid:
#         return {
#             "messages": [
#                 SystemMessage(
#                     content="The SPARQL results have been validated and appear correct. Please formulate your final answer to the user based on these results."
#                 )
#             ]
#         }
#     else:
#         content = (
#             f"The previous query returned results, but the automated validator determined "
#             f"they do not correctly answer the question: '{original_question}'.\n"
#             f"Validator Feedback: {feedback_text}\n"
#             f"Please analyze the previous query and results, and generate a CORRECTED SPARQL query."
#         )
#         return {
#             "messages": [SystemMessage(content=content)]
#         }
