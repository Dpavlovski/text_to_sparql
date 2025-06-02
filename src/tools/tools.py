import json
from typing import Any, Optional, Dict, List

from pydantic import BaseModel
from qdrant_client.http.models import ScoredPoint
from rdflib.plugins.sparql import prepareQuery

from src.databases.qdrant.search_embeddings import fetch_similar_entities, search_embeddings
from src.llm.generic_chat import generic_chat
from src.templates.ner import extract_entities
from src.templates.sparql import sparql_template
from src.utils.json__extraction import get_json_response
from src.wikidata.api import execute_sparql_query
from src.wikidata.dump_processing.test import get_entities_schema, format_schema_output


class ToolResult(BaseModel):
    success: bool
    payload: Optional[Any]
    error: Optional[str]


def safe_tool(fn):
    def wrapper(inputs: Any) -> ToolResult:
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except json.JSONDecodeError:
                pass
        try:
            result = fn(inputs)
            return ToolResult(success=True, payload=result, error=None)
        except Exception as e:
            return ToolResult(success=False, error=f"{fn.__name__} failed: {e}", payload=None)

    return wrapper


@safe_tool
def tool_get_ner_results(inputs: Dict[str, Any]) -> Any:
    return extract_entities(inputs["question"])


@safe_tool
def tool_fetch_similar_entities(inputs: Dict[str, Any]) -> list[ScoredPoint]:
    labels = inputs["labels"]
    lang = inputs.get("lang", "en")
    return fetch_similar_entities(labels, lang)


@safe_tool
def get_schema(entities: List[Any]):
    schema = get_entities_schema(entities)
    return format_schema_output(schema, entities)


@safe_tool
def tool_get_examples(inputs: Dict[str, Any]) -> Any:
    question = inputs["question"]
    return search_embeddings(question, collection_name="lcquad2_0")


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


@safe_tool
def tool_generate_sparql(inputs: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(inputs, str):
        inputs = json.loads(inputs)
    question = inputs["question"]
    examples = inputs["examples"]
    sim_entities = inputs["similar_entities"]

    sparql_prompt = sparql_template(question, examples, sim_entities)
    generated = get_json_response(
        sparql_prompt,
        list_name="sparql",
        system_message="You are a SPARQL query generator."
    )
    sparql = generated.get("sparql")
    if not sparql:
        raise ValueError("Missing 'sparql' in generation response")

    is_valid, msg = validate_sparql(sparql)
    if not is_valid:
        raise ValueError(f"Invalid SPARQL: {sparql} {msg}")

    return {"sparql": sparql}


@safe_tool
def tool_execute_sparql(inputs: Dict[str, Any]) -> Dict[str, Any]:
    query = inputs["sparql"]
    results = execute_sparql_query(query)
    return {"query": query, "results": results}


@safe_tool
def tool_error_handling(inputs: Dict[str, Any]) -> Dict[str, Any]:
    error_msg = inputs.get("error", "Unknown error")
    llm_response = generic_chat(
        message=f"Tool error occurred. Please analyze and suggest next steps. Error: {error_msg}"
    )
    return {"message": llm_response}
