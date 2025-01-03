import os
from enum import Enum

from dotenv import load_dotenv

from text_to_sparql.llm.chat import chat_with_ollama
from text_to_sparql.llm.gpt_chat import chat_with_openai


class ChatModel(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


def generic_chat(message: str, system_message: str = "You are a helpful AI assistant.") -> str:
    load_dotenv()
    chat_model = os.getenv("CHAT_MODEL")

    if chat_model == ChatModel.OPENAI.value:
        return chat_with_openai(message, system_message)
    elif chat_model == ChatModel.OLLAMA.value:
        return chat_with_ollama(message, system_message)
    else:
        raise ValueError("CHAT_MODEL environment variable is not set or has an invalid value.")
