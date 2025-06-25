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


def search_wikidata(keyword, type, lang="en"):
    results = []
    params = {
        "action": "wbsearchentities",
        "search": keyword,
        "type": type,
        "language": lang,
        "format": "json",
        "limit": 5
    }
    wikidata_result = fetch_wikidata(params)
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

        return []

    except Exception as e:
        print(f"An error occurred while processing the SPARQL response: {e}")
        raise
