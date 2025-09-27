from src.llm.embed_labels import embed_value


def embed_examples(
        content: str
) -> list[float] | None:


    try:
        return embed_value(content)
    except Exception as e:
        print(f"An error occurred: {e}")
