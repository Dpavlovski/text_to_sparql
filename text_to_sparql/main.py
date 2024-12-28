from text_to_sparql.llm.chat import chat_with_ollama
from text_to_sparql.templates.ner import extract_entities_and_relations
from text_to_sparql.templates.sparql import sparql_template
from text_to_sparql.utils.format_results import format_results
from text_to_sparql.wikidata.api import search_wikidata


def main():
    question = "What are Einstein research fields in physics?"

    ner_results = extract_entities_and_relations(question)
    entities = ner_results.get("entities", [])
    relations = ner_results.get("relations", [])

    print(f"Extracted Entities: {entities}")
    print(f"Extracted Relations: {relations}")

    entity_results = search_wikidata(entities, "item")
    relation_results = search_wikidata(relations, "property")

    entity_descriptions, relations_descriptions = format_results(entity_results, relation_results)

    print(f"Entity Description: {entity_descriptions}")
    print(f"Relation Description: {relations_descriptions}")

    sparql_prompt = sparql_template(question, entity_descriptions, relations_descriptions)
    sparql_query = chat_with_ollama(sparql_prompt, system_message="You are a SPARQL query generator.")
    print("Generated SPARQL Query:", sparql_query)


if __name__ == "__main__":
    main()
