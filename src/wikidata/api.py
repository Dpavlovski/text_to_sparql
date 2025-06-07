import json
from typing import Any

import requests
from SPARQLWrapper import SPARQLWrapper, JSON


def fetch_wikidata(params) -> str | Any:
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


def search_wikidata(keywords, keyword_type):
    results = []
    for keyword in keywords:
        params = {
            "action": "wbsearchentities",
            "search": keyword,
            "language": "en",
            "type": keyword_type,
            "format": "json",
            "limit": 5
        }
        wikidata_result = fetch_wikidata(params)
        # print(f"Wikidata Results for {keyword_type} '{keyword}':", wikidata_result)
        if isinstance(wikidata_result, dict):
            results.extend(wikidata_result["search"])
    return results


def execute_sparql_query(query):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        responses = sparql.query().convert()

        if "results" in responses and "bindings" in responses["results"]:
            bindings = responses["results"]["bindings"]
            if bindings:
                return [
                    {value["value"] for var, value in result.items()} for result in bindings
                ]

        elif "boolean" in responses:
            return responses["boolean"]

        return None

    except Exception as e:
        print(f"An error occurred while processing the SPARQL response: {e}")
        raise

# def fetch_wikidata_labels(entity_ids: List[str]) -> Dict[str, List[dict]]:
#     logging.basicConfig(level=logging.INFO)
#     logging.info("Fetching labels for given Wikidata entity URIs...")
#
#     client = Client()
#     labels_map: Dict[str, List[dict]] = {}
#
#     for entity_id in tqdm(entity_ids, desc="Fetching labels"):
#         try:
#             entity = client.get(entity_id, load=True)
#
#             while hasattr(entity, 'id') and entity.id != entity_id:
#                 logging.info(f"{entity_id} is a redirect. Redirecting to {entity.id}")
#                 entity_id = entity.id
#                 entity = client.get(entity_id, load=True)
#
#             if len(entity.label) == 0:
#                 logging.info(f"{entity_id} is missing a label. Skipping.")
#                 continue
#
#             label_entries = []
#             for lang, label in entity.label.items():
#                 description = entity.description.get(lang, "") if hasattr(entity, 'description') else ""
#                 label_entries.append({
#                     "label": label,
#                     "description": description,
#                     "language": lang
#                 })
#
#             labels_map[entity_id] = label_entries
#         except Exception as e:
#             logging.error(f"Error fetching label for {entity_id}: {e}")
#             labels_map[entity_id] = []
#
#     return labels_map
