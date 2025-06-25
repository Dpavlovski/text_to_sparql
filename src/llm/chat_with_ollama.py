import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


def chat_with_ollama() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL"),
        base_url=os.getenv("OLLAMA_API_URL"),
        temperature=0,
    )
