import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def chat_with_openai(
) -> ChatOpenAI:
    load_dotenv()

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        base_url=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )
