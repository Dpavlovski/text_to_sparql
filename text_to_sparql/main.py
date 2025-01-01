from text_to_sparql.databases.qdrant.qdrant import QdrantDatabase
from text_to_sparql.llm.chat import chat_with_ollama
from text_to_sparql.templates.ner import extract_and_refine_wikidata_keywords
from text_to_sparql.templates.sparql import sparql_template
from text_to_sparql.utils.format_results import format_results
from text_to_sparql.wikidata.api import search_wikidata, execute_sparql_query


def main():
    question = "What kind of research was Einstein involved in Physics?"

    examples = QdrantDatabase().search_embeddings_str(question, score_threshold=0.2, top_k=5,
                                                      collection_name="lcquad2_0")

    ner_results = extract_and_refine_wikidata_keywords(question)
    entities = ner_results.get("entities", [])
    relations = ner_results.get("relations", [])

    entity_results = search_wikidata(entities, "item")
    relation_results = search_wikidata(relations, "property")

    entity_descriptions, relations_descriptions = format_results(entity_results, relation_results)

    print(f"Entity Description: {entity_descriptions}")
    print(f"Relation Description: {relations_descriptions}")

    sparql_prompt = sparql_template(question, examples, entity_descriptions, relations_descriptions)
    print("SPARQL Prompt:")
    print(sparql_prompt)

    sparql_query = chat_with_ollama(sparql_prompt, system_message="You are a SPARQL query generator.")
    print("Generated SPARQL Query:")
    print(sparql_query)

    print("Results of the SPARQL Query:")
    execute_sparql_query(sparql_query)


if __name__ == "__main__":
    main()
