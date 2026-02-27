from typing import List

from qdrant_client.http.models import ScoredPoint


def format_qa_sparql_examples(examples: List[ScoredPoint]) -> str:
    """
    Formats dynamic few-shot examples to teach the LLM how to map
    a natural language question to a SPARQL query.
    """
    if not examples:
        return ""

    formatted_examples = []
    for ex in examples:
        # Using .get() is safer than direct access and avoids try/except blocks
        payload = ex.payload
        if not payload:
            continue

        question = payload.get("value")
        sparql_query = payload.get("answer")

        # We only include the example if both the question and SPARQL are present
        if question and sparql_query:
            example_str = (
                f"Question: {question}\n"
                f"SPARQL: {sparql_query}"
            )
            formatted_examples.append(example_str)

    # Join the examples with a clear separator for the LLM
    if not formatted_examples:
        return ""

    return "Here are some examples of how to translate a question into a SPARQL query:\n\n---\n" + \
        "\n\n---\n".join(formatted_examples) + \
        "\n\n---"
