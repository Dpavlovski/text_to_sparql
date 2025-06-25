# import json
# import os
# import uuid
# from typing import List, Dict, Any
#
# from langchain import hub
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_core.tools import Tool
# from langchain_ollama import OllamaLLM
#
# from src.databases.mongo.models.AgentStep import AgentStep
# from src.databases.mongo.mongo import MongoDBDatabase
# from src.tools.tools import (
#     tool_get_ner_results,
#     tool_fetch_similar_entities,
#     tool_get_examples,
#     tool_generate_sparql,
#     tool_execute_sparql, tool_error_handling, )
#
#
# def initialize_tools() -> List[Tool]:
#     return [
#         Tool(
#             name="Extract_Labels",
#             func=tool_get_ner_results,
#             description="""
#             inputs: {"question": str} → returns {"labels": [...], "lang": str}
#             Extracts all relevant keywords and phrases (entities, nouns, verbs) from the question, plus detects language.
#             """
#         ),
#         Tool(
#             name="Find_Similar_Entities",
#             func=tool_fetch_similar_entities,
#             description="""
#             inputs: {"labels": [...], "lang": str} → returns list of scored Wikidata entities
#             """
#         ),
#         Tool(
#             name="Find_Similar_Examples",
#             func=tool_get_examples,
#             description="""
#             inputs: {"question": str} → returns nearest example question-SPARQL pairs
#             """
#         ),
#         Tool(
#             name="Generate_SPARQL",
#             func=tool_generate_sparql,
#             description="""
#             inputs: {"question": str, "examples": [...], "similar_entities": [...]} → returns {"sparql": str}
#             """
#         ),
#         Tool(
#             name="Execute_SPARQL",
#             func=tool_execute_sparql,
#             description="""
#             inputs: {"sparql": str} → returns {"sparql": str, "results": [...]}}
#             """
#         ),
#         Tool(
#             name="Error_Handling",
#             func=tool_error_handling,
#             description="""
#             Handles errors from other tools. Input: {'error': 'description'}
#             """
#         ),
#     ]
#
#
# def format_final_response(
#         self,
#         raw_response: Dict[str, Any],
#         question: str,
#         run_id: str
# ) -> Dict[str, Any]:
#     steps = raw_response.get("intermediate_steps", [])
#
#     for i, (tool_call, output) in enumerate(steps):
#         try:
#             if isinstance(tool_call.tool_input, str):
#                 tool_input = json.loads(tool_call.tool_input)
#             else:
#                 tool_input = AgentStep.serialize_value(tool_call.tool_input)
#         except json.JSONDecodeError:
#             tool_input = {"raw_input": tool_call.tool_input}
#
#         try:
#             tool_output = AgentStep.serialize_value(output)
#         except Exception as e:
#             tool_output = {"error": f"Serialization failed: {str(e)}", "raw_output": str(output)}
#
#         step = AgentStep(
#             run_id=run_id,
#             question=question,
#             step_number=i,
#             tool_name=tool_call.tool,
#             tool_input=tool_input,
#             tool_output=tool_output
#         )
#         self.db.add_entry(step, collection_name="AgentSteps")
#
#     if "output" in raw_response:
#         return {
#             "answer": raw_response["output"],
#             "run_id": run_id,
#             "steps": len(steps)
#         }
#     return {
#         "error": "No output generated",
#         "run_id": run_id,
#         "steps": len(steps)
#     }
#
#
# class SPARQLAgent:
#     def __init__(self):
#         self.llm = OllamaLLM(
#             model=os.getenv("OLLAMA_MODEL"),
#             base_url="https://llama3.finki.ukim.mk",
#             temperature=0,
#         )
#         self.tools = initialize_tools()
#         self.db = MongoDBDatabase()
#         self.agent_executor = self._create_agent()
#
#     def _create_agent(self) -> AgentExecutor:
#         react_prompt = hub.pull("hwchase17/react")
#         agent = create_react_agent(
#             llm=self.llm,
#             tools=self.tools,
#             prompt=react_prompt
#         )
#         return AgentExecutor(
#             agent=agent,
#             tools=self.tools,
#             handle_parsing_errors=True,
#             verbose=True,
#             return_intermediate_steps=True,
#         )
#
#     def process_question(self, question: str) -> Dict[str, Any]:
#         run_id = str(uuid.uuid4())
#
#         response = self.agent_executor.invoke(
#             {"input": question},
#         )
#         return format_final_response(self, response, question, run_id)
