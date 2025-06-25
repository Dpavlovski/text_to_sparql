from typing import List, Any, Dict

from rdflib.plugins.sparql import prepareQuery

from src.utils.format_results import format_results
from src.utils.json__extraction import get_json_response
from src.wikidata.api import execute_sparql_query


def sparql_template(question: str, examples: str, items: List[Any], properties: List[Any]) -> str:
    return f"""You are an AI designed to generate precise SPARQL queries for retrieving information from the Wikidata knowledge graph. 

Your task:
- Use only the provided entities to construct the query.
- Do not include prefixes or services.
- Use only "wd" or "wdt" as prefixes for entities.
- Determine whether a "SELECT" or "ASK" query is more appropriate.
- Return only the SPARQL query in JSON format with the key 'sparql', without any additional text.

{examples}`

Question: {question}

{format_results(items, properties)}
Output format (JSON):
{{
  "sparql": "<SPARQL_QUERY_HERE>"
}}
"""


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


def get_sparql_query(question: str, examples: str, sim_entities: Dict[str, Any]) -> dict[str, Any]:
    sparql_prompt = sparql_template(question, examples, sim_entities.get("items", []),
                                    sim_entities.get("properties", []))
    sparql = get_json_response(
        sparql_prompt,
        list_name="sparql",
        system_message="You are a SPARQL query generator."
    )
    if not sparql:
        raise ValueError("Missing 'sparql' in generation response")

    is_valid, msg = validate_sparql(sparql["sparql"])
    if not is_valid:
        raise ValueError(f"Invalid SPARQL: {sparql} {msg}")

    results = execute_sparql_query(sparql["sparql"])
    return {"sparql": sparql, "results": results}
