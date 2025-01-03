from typing import List

from qdrant_client.http.models import ScoredPoint

from text_to_sparql.utils.format_examples import format_examples


def sparql_template(question: str, examples: List[ScoredPoint], entity_descriptions: str, relations_descriptions: str):
    formatted_examples = f"Examples:\n{format_examples(examples)}" if examples else ""
    entity_descriptions = f"Entities:\n{entity_descriptions}" if entity_descriptions else ""
    relations_descriptions = f"Relations:\n{relations_descriptions}" if relations_descriptions else ""

    return f"""You are an AI that generates precise SPARQL queries to answer the given question. 
Your task is to carefully select the most relevant entities and relations from the provided options in order to answer the question. Use only the provided entities and relations.
You must only return the sparql query and nothing else.

{formatted_examples}

Question: {question}

{entity_descriptions}

{relations_descriptions}

Return in json format with key 'sparql'.
"""
