from typing import List, Dict

from qdrant_client.http.models import ScoredPoint

from src.databases.qdrant.search_embeddings import fetch_similar_entities, extract_search_objects
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


def get_neighbors(node_id: str, hops: int = 1, limit: int = 5, direction: str = "both") -> List[Dict]:
    if not node_id.upper().startswith(("Q", "P")):
        raise ValueError("Invalid node ID. Must start with Q or P")

    wd_entity = f"wd:{node_id.upper()}"

    direction_filters = []
    if direction in ("out", "both"):
        direction_filters.append(f"?node ?p ?neighbor .")
    if direction in ("in", "both"):
        direction_filters.append(f"?neighbor ?p ?node .")

    query = f"""
        SELECT DISTINCT ?neighbor
        WHERE {{
            VALUES ?node {{ {wd_entity} }}

            {{
                ?node {" ?p1 ?neighbor1 . " if hops >= 1 else ""}
                {" ?neighbor1 ?p2 ?neighbor2 . " if hops >= 2 else ""}
                {" ?neighbor2 ?p3 ?neighbor3 . " if hops >= 3 else ""}
            }}
            BIND({f"?neighbor{hops}" if hops > 0 else "?node"} AS ?neighbor)
            FILTER(?neighbor != {wd_entity}) 
        }}
        LIMIT {limit}
    """

    results = execute_sparql_query(query)
    return results


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
                for value in row:
                    print(value)
                    answers.append(value)
                print()
        elif isinstance(initial_result, bool):
            print(initial_result)
            return initial_sparql, initial_result
        return initial_sparql, answers

    else:
        examples = extract_search_objects(question, collection_name="lcquad2_0")

        tries = 3
        for i in range(tries):
            print(f"Attempt {i + 1}/{tries} to generate and execute SPARQL query.")

            try:
                ner_results = get_ner_results(question)

                similar_entities = fetch_similar_entities(ner_results["labels"], ner_results["lang"])

                # neighbors = get_neighbors(similar_entities[0].payload.get("id"), hops=1, limit=5)
                # print("Neighbors:")
                # print(neighbors)

                print("Generated SPARQL Queries:")
                query, result = perform_multi_querying_with_ranking(
                    question, examples, similar_entities,
                )

                answers = []

                if isinstance(result, list):
                    print("Selected Query:")
                    print(query)
                    print("\nResults:")

                    for row in result:
                        for value in row:
                            answer = value
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
    question = "Which country was Bill Gates born in?"

    text_to_sparql(question)


if __name__ == "__main__":
    main()
