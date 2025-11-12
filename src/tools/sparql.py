from typing import List, Any, Dict

from rdflib.plugins.sparql import prepareQuery

from src.agent.prompts import sparql_prompt_template, disambiguation_prompt_template
from src.utils.json__extraction import get_json_response
from src.wikidata.api import execute_sparql_query
from src.wikidata.prefixes import PREFIXES


def format_candidates_for_disambiguation(candidates_map: Dict[str, List[Dict[str, Any]]]) -> str:
    """Formats the candidate map into a readable string for the prompt."""
    output = ""
    for mention, candidates in candidates_map.items():
        output += f"For the mention '{mention}':\n"
        if not candidates:
            output += "  - No candidates found.\n"
            continue
        for i, cand in enumerate(candidates):
            output += f"  {i + 1}. ID: {cand['id']}, Label: {cand['label']}, Description: {cand['description']}\n"
        output += "\n"
    return output.strip()


def format_linked_entities_for_prompt(linked_entities: Dict[str, str]) -> str:
    """Formats the confirmed entities into a readable string for the prompt."""
    return "\n".join(
        f"- The entity for the mention '{mention}' is `{entity_id}`."
        for mention, entity_id in linked_entities.items()
    )


async def disambiguate_entities(
        question: str,
        candidates_map: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, str]:
    """
    Takes a question and a map of ambiguous candidates and returns a map of confirmed entity IDs.
    """
    if not candidates_map:
        return {}

    formatted_candidates = format_candidates_for_disambiguation(candidates_map)

    prompt = disambiguation_prompt_template.format(
        question=question,
        formatted_candidates=formatted_candidates
    )

    linked_entities = await get_json_response(prompt)

    if not isinstance(linked_entities, dict):
        raise ValueError("Disambiguation response was not a valid dictionary.")

    return linked_entities


def validate_sparql(query: str):
    try:
        full_query = PREFIXES + query
        prepareQuery(full_query)
        return True, "Query is valid."
    except Exception as e:
        return False, f"Validation error: {e}"


async def get_sparql_query(
        question: str,
        examples: str,
        linked_entities: Any
) -> Dict[str, Any]:
    linked_entities_context = format_linked_entities_for_prompt(linked_entities)

    sparql_prompt = sparql_prompt_template.format(
        examples=examples,
        question=question,
        linked_entities_context=linked_entities_context
    )

    sparql_json = await get_json_response(sparql_prompt)

    if not sparql_json or "sparql" not in sparql_json:
        raise ValueError("Missing 'sparql' in generation response")

    sparql_query_str = sparql_json["sparql"]
    is_valid, msg = validate_sparql(sparql_query_str)
    results = None

    if is_valid:
        results = await execute_sparql_query(sparql_query_str)

    return {"sparql": sparql_query_str, "results": results}
