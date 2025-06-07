from typing import List, Dict, Optional, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The initial user question.
        lang: The detected language of the question.
        labels: Extracted entities and keywords from the question.
        similar_entities: Wikidata entities found to be similar to the labels.
        examples: Similar question-SPARQL pairs from the example database.
        sparql_query: The generated SPARQL query.
        results: The data returned from executing the SPARQL query.
        error: A string to hold any error messages.
        next_action: The next tool the agent should call.
        final_answer: The final, user-facing answer.
"""
    question: str
    lang: Optional[str] = None
    labels: Optional[List[str]] = None
    similar_entities: Optional[List[Dict]] = None
    examples: Optional[List[Dict]] = None
    sparql_query: Optional[str] = None
    results: Optional[List[Dict]] = None
    error: Optional[str] = None
    next_action: Optional[str] = None
    final_answer: Optional[str] = None
