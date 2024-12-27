import json

import requests

from text_to_sparql.chat import chat_with_ollama


def fetch_wikidata(params):
    url = 'https://www.wikidata.org/w/api.php'
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"HTTP error occurred: {e}"
    except json.JSONDecodeError:
        return "Error decoding JSON"
    except Exception as e:
        return f"An error occurred: {e}"


def extract_entities_and_relations(question):
    ner_prompt = f"""
        You are an AI that extracts entities and relations from a given question. 
        Your task is to analyze the question and identify:
        - **Entities**: Specific people, places, items, concepts, or other relevant subjects mentioned in the question.
        - **Relations**: Actions, associations, or connections described in the question.

        Your response must strictly adhere to the following JSON format:
        {{
            "entities": ["entity1", "entity2", ...],
            "relations": ["relation1", "relation2", ...]
        }}

        Make sure to:
        - Only include valid entities and relations extracted from the question.
        - Avoid adding explanations, comments, or additional text outside the JSON structure.

        **Examples:**

        1. **Input Question**: "Who is the president of the United States?"
           **Output**:
           {{
               "entities": ["president", "United States"],
               "relations": ["is"]
           }}

        2. **Input Question**: "Where was Edison born?"
           **Output**:
           {{
               "entities": ["Edison"],
               "relations": ["was born"]
           }}

        3. **Input Question**: "What are the capitals of European countries?"
           **Output**:
           {{
               "entities": ["capitals", "European countries"],
               "relations": ["are"]
           }}

        Now, analyze the following question and return the extracted entities and relations in the required JSON format:

        Question: {question}
        """
    response = chat_with_ollama(ner_prompt, system_message="You are an NER assistant.")
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Error decoding the NER response. Ensure the response is in valid JSON format.")
        return {"entities": [], "relations": []}


def search_wikidata(keywords, keyword_type):
    results = []
    for keyword in keywords:
        params = {
            "action": "wbsearchentities",
            "search": keyword,
            "type": keyword_type,
            "language": "en",
            "format": "json",
            "limit": 5
        }
        wikidata_result = fetch_wikidata(params)
        print(f"Wikidata Results for {keyword_type} '{keyword}':", wikidata_result)
        if isinstance(wikidata_result, dict) and "search" in wikidata_result:
            results.extend(wikidata_result["search"])
    return results


def main():
    question = "What did Einstein discover in physics?"

    ner_results = extract_entities_and_relations(question)
    entities = ner_results.get("entities", [])
    relations = ner_results.get("relations", [])

    print(f"Extracted Entities: {entities}")
    print(f"Extracted Relations: {relations}")

    entity_results = search_wikidata(entities, "entity")
    relation_results = search_wikidata(relations, "property")

    entity_descriptions = "\n".join(
        [f"Entity: {e['label']} (ID: {e['id']}) - {e.get('description', 'No description available')}"
         for e in entity_results]
    )

    relations_descriptions = "\n".join(
        [f"Relation: {e['label']} (ID: {e['id']}) - {e.get('description', 'No description available')}"
         for e in relation_results]
    )

    print(f"Entity Description: {entity_descriptions}")
    print(f"Relation Description: {relations_descriptions}")

    # sparql_prompt = f"""
    # You are an AI that generates SPARQL queries based on the question and the provided entities and their relations.
    # Entities:
    # {entity_descriptions}
    #
    # Generate a SPARQL query to retrieve related data for these entities.
    # """
    #
    # sparql_query = chat_with_ollama(sparql_prompt, system_message="You are a SPARQL query generator.")
    # print("Generated SPARQL Query:", sparql_query)


if __name__ == "__main__":
    main()
