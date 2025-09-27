from typing import List, Any, Dict

from rdflib.plugins.sparql import prepareQuery

from src.agent.prompts import sparql_prompt_template
from src.utils.format_entities import format_entities
from src.utils.format_results import format_results
from src.utils.json__extraction import get_json_response
from src.wikidata.api import execute_sparql_query


def sparql_template(question: str, examples: str, items: List[Any], properties: List[Any],
                    embeddings: List[Any]) -> str:
    formatted_examples = f"Examples:\n{examples}" if examples else ""
    entity_descriptions, relations_descriptions = format_results(items, properties)
    entity_descriptions = f"Entities:\n{entity_descriptions}" if entity_descriptions else ""
    relations_descriptions = f"Relations:\n{relations_descriptions}" if relations_descriptions else ""

    return sparql_prompt_template.format(
        formatted_examples=formatted_examples,
        question=question,
        entity_descriptions=entity_descriptions,
        relations_descriptions=relations_descriptions,
        embeddings=format_entities(embeddings) if embeddings else ""
    )


def validate_sparql(query: str):
    try:
        prefixes = """
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        """

        full_query = prefixes + query
        prepareQuery(full_query)
        return True, "Query is valid."
    except Exception as e:
        return False, f"Validation error: {e}"


async def get_sparql_query(question: str, examples: str, sim_entities: Dict[str, Any]) -> dict[str, Any]:
    sparql_prompt = sparql_template(question, examples, sim_entities.get("items", []),
                                    sim_entities.get("properties", []), sim_entities.get("embeddings", []))

    sparql = await get_json_response(
        sparql_prompt,
        list_name="sparql"
    )

    if not sparql or "sparql" not in sparql:
        raise ValueError("Missing 'sparql' in generation response")

    is_valid, msg = validate_sparql(sparql["sparql"])
    results = None

    if is_valid:
        results = await execute_sparql_query(sparql["sparql"])

    return {"sparql": sparql["sparql"], "results": results}
