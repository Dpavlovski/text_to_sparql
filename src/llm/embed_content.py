from typing import List

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_ollama import OllamaEmbeddings


def embed_content(
        content: str
) -> List[float]:
    model = "llama3.1:70b"
    ollama_embeddings = OllamaEmbeddings(
        model=model,
        base_url="https://llama3.finki.ukim.mk",
    )

    store = LocalFileStore("./cache/")

    embeddings = CacheBackedEmbeddings.from_bytes_store(
        ollama_embeddings, store, namespace="fcse"
    )

    try:
        return embeddings.embed_query(content)
    except Exception as e:
        print(f"An error occurred: {e}")
