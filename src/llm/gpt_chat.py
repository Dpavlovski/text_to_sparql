import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def chat_with_openai(
) -> ChatOpenAI:
    load_dotenv()

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
    )

# print(chat_with_openai().invoke("Hello"))
