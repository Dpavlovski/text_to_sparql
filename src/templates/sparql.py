from typing import List

from qdrant_client.http.models import ScoredPoint

from src.utils.format_entities import format_entities
from src.utils.format_examples import format_examples


def sparql_template(question: str, examples: List[ScoredPoint], similar_entities: List[ScoredPoint]) -> str:
    return f"""You are an AI designed to generate precise SPARQL queries for retrieving information from the Wikidata knowledge graph. 

Your task:
- Use only the provided entities to construct the query.
- Do not include prefixes or services.
- Use only "wd" or "wdt" as prefixes for entities.
- Determine whether a "SELECT" or "ASK" query is more appropriate.
- Return only the SPARQL query in JSON format with the key 'sparql', without any additional text.

{format_examples(examples)}`

Question: {question}

{format_entities(similar_entities)}
Output format (JSON):
{{
  "sparql": "<SPARQL_QUERY_HERE>"
}}
"""
