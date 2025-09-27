import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

from dotenv import load_dotenv

load_dotenv()


def chat_with_ollama():
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL"),
        base_url=os.getenv("OLLAMA_API_URL"),
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"},
            "timeout": 60,
        },
        temperature=0,
    )

# print(chat_with_ollama().invoke("Hello").content)
