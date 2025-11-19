# import logging
# from typing import Any, Dict
#
# from src.llm.generic_llm import generic_llm
# from src.utils.trim_and_load_json import trim_and_load_json
#
#
# async def get_json_response(
#         template: str,
#         list_name: str = ""
# ) -> Dict[str, Any]:
#     llm = generic_llm()
#     full_response_content = ""
#
#     try:
#         # This is where the call to the Ollama model happens.
#         # We'll wrap it in a try...except block.
#         async for chunk in llm.astream(template):
#             full_response_content += chunk.content
#
#     except Exception as e:
#         # If an error occurs during the Ollama call, this block will execute.
#         print("=" * 80)
#         print(f"ERROR: An exception occurred while calling the Ollama language model.")
#         print(f"       This is the likely source of the timeout or connection error.")
#         print(f"       Error details: {e}")
#         print("=" * 80)
#         # Return an empty dictionary to handle the error gracefully
#         return {}
#
#     is_finished, json_data = trim_and_load_json(
#         input_string=full_response_content,
#         list_name=list_name
#     )
#
#     if not is_finished:
#         logging.warning("Failed to parse a valid JSON from the streamed response.")
#         return {}
#
#     return json_data
