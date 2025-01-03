from typing import List

from qdrant_client.http.models import ScoredPoint

from text_to_sparql.databases.qdrant.search_embeddings import extract_search_objects
from text_to_sparql.templates.ner import ner_template
from text_to_sparql.templates.sparql import sparql_template
from text_to_sparql.utils.format_results import format_results
from text_to_sparql.utils.json__extraction import get_json_response
from text_to_sparql.wikidata.api import search_wikidata, execute_sparql_query


def get_ner_results(question: str) -> dict:
    return get_json_response(
        ner_template(question),
        list_name="nodes",
        system_message="You are a Wikidata entity and relation extraction assistant."
    )


def get_sparql(question: str, examples: List[ScoredPoint], entity_descriptions: str,
               relations_descriptions: str) -> dict:
    sparql_prompt = sparql_template(question, examples, entity_descriptions, relations_descriptions)

    return get_json_response(sparql_prompt, list_name="sparql",
                             system_message="You are a SPARQL query generator.")


def main():
    question = "What kind of research was Einstein involved in Physics?"

    examples = extract_search_objects(question, collection_name="lcquad2_0")

    ner_results = get_ner_results(question)

    entities = [result['wikidata_label'] for result in ner_results["nodes"] if result['wikidata_type'] == "item"]
    relations = [result['wikidata_label'] for result in ner_results["nodes"] if result['wikidata_type'] == "property"]

    entity_results = search_wikidata(entities, "item")
    relation_results = search_wikidata(relations, "property")

    entity_descriptions, relations_descriptions = format_results(entity_results, relation_results)

    print(f"Entity Description: {entity_descriptions}")
    print(f"Relation Description: {relations_descriptions}")

    print("Examples:")
    print(examples)

    sparql_query = get_sparql(question, examples, entity_descriptions, relations_descriptions)["sparql"]
    print("Generated SPARQL Query:")
    print(sparql_query)

    print("Results of the SPARQL Query:")
    execute_sparql_query(sparql_query)


if __name__ == "__main__":
    main()
