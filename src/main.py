from typing import Any

from src.agent.agent import SPARQLAgent


def text_to_sparql(question: str) -> tuple[None, list[Any]] | tuple[Any, list[Any]]:
    agent = SPARQLAgent()
    result = agent.process_question(question)

    if result.get("error"):
        print(f"Error: {result['error']}")
        return None, []

    if isinstance(result.get("results"), bool):
        return result["query"], [result["results"]]

    return None, []


def main():
    question = "What is the boiling point of water?"
    text_to_sparql(question)


if __name__ == "__main__":
    main()