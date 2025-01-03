import logging
from typing import Any, Dict

from pydantic import BaseModel

from text_to_sparql.llm.generic_chat import generic_chat
from text_to_sparql.utils.trim_and_load_json import trim_and_load_json


class ChatResponse(BaseModel):
    message: str
    response: str
    llm_model: str


def get_json_response(
        template: str,
        list_name: str = "",
        system_message: str = "You are a helpful AI assistant."
) -> Dict[str, Any]:
    is_finished = False
    json_data = {}
    tries = 0

    while not is_finished:
        if tries > 0:
            logging.warning(f"Chat not returning as expected. Attempt: {tries}")

        if tries > 3:
            logging.error("Exceeded maximum retry attempts for chat response.")
            raise Exception("Chat model failed to return a valid response.")

        response = generic_chat(message=template, system_message=system_message)

        is_finished, json_data = trim_and_load_json(input_string=response, list_name=list_name)
        tries += 1

    return json_data
