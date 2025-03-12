from typing import List

from qdrant_client.http.models import ScoredPoint

from src.databases.qdrant.search_embeddings import fetch_similar_entities
from src.templates.ner import ner_template
from src.templates.sparql import sparql_template
from src.templates.zero_shot_sparql import zero_shot_sparql
from src.utils.json__extraction import get_json_response
from src.wikidata.api import execute_sparql_query


def get_ner_results(question: str) -> dict:
    return get_json_response(
        ner_template(question),
        list_name="labels",
        system_message="You are a Wikidata entity extraction assistant."
    )


def get_sparql(question: str, examples: List[ScoredPoint], similar_entities: List[ScoredPoint]) -> dict:
    sparql_prompt = sparql_template(question, examples, similar_entities)

    return get_json_response(sparql_prompt, list_name="sparql",
                             system_message="You are a SPARQL query generator.")["sparql"]


def perform_multi_querying_with_ranking(question, examples, similar_entities, ):
    sparql_query = None
    for i in range(5):
        try:
            sparql_query = get_sparql(question, examples, similar_entities)
            print(f"Query {i + 1}: {sparql_query}")

            results = execute_sparql_query(sparql_query)

            if results is not None:
                return sparql_query, results
            else:
                print("No results found.")
        except Exception as e:
            print(f"Error during query {i + 1}: {e}")

    return sparql_query, None


def text_to_sparql(question):
    answers = []
    query = None
    initial_sparql = None
    initial_result = None

    try:
        initial_sparql = get_json_response(zero_shot_sparql(question), list_name="sparql",
                                       system_message="You are a SPARQL query generator.")["sparql"]
        print(initial_sparql)
        initial_result = execute_sparql_query(initial_sparql)
    except Exception as e:
        print(f"Error during query: {e}")

    if initial_result is not None:
        print("Query:")
        print(initial_sparql)
        print("Result:")
        if isinstance(initial_result, list):
            for row in initial_result:
                for var, value in row.items():
                    print(f"{var}: {value['value']}")
                    answers.append(f"{var}: {value['value']}")
                print()
        elif isinstance(initial_result, bool):
            print(initial_result)
            return initial_sparql, initial_result
        return initial_sparql, answers

    else:
        # examples = extract_search_objects(question, collection_name="lcquad2_0")

        tries = 3
        for i in range(tries):
            print(f"Attempt {i + 1}/{tries} to generate and execute SPARQL query.")

            try:
                ner_results = get_ner_results(question)

                similar_entities = fetch_similar_entities(ner_results["labels"], ner_results["lang"])

                print("Generated SPARQL Queries:")
                query, result = perform_multi_querying_with_ranking(
                    question, None, similar_entities,
                )

                answers = []

                if isinstance(result, list):
                    print("Selected Query:")
                    print(query)
                    print("\nResults:")

                    for row in result:
                        for var, value in row.items():
                            answer = f"{var}: {value['value']}"
                            print(answer)
                            answers.append(answer)
                        print()

                    return query, answers

                elif isinstance(result, bool):
                    print("Selected Query:")
                    print(query)
                    print("\nResult:")
                    print(result)

                    return query, [result]

            except Exception as e:
                print(f"An error occurred during attempt {i + 1}: {e}")

        else:
            print("All attempts failed. No valid SPARQL query returned results.")
            return query, []


def main():
    question = ""

    text_to_sparql(question)


if __name__ == "__main__":
    main()
