import json

import requests
from SPARQLWrapper import SPARQLWrapper, JSON


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
        print(f"Wikidata Results for {keyword_type} '{keyword}':", wikidata_result)
        if isinstance(wikidata_result, dict) and "search" in wikidata_result:
            results.extend(wikidata_result["search"])
    return results


def execute_sparql_query(query):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()

        for result in results["results"]["bindings"]:
            predicate1 = result.get("predicate1", {}).get("value", "N/A")
            neighbor1 = result.get("neighbor1", {}).get("value", "N/A")
            neighbor1_label = result.get("neighbor1Label", {}).get("value", "N/A")
            predicate2 = result.get("predicate2", {}).get("value", "N/A")
            neighbor2 = result.get("neighbor2", {}).get("value", "N/A")
            neighbor2_label = result.get("neighbor2Label", {}).get("value", "N/A")

            print(f"Predicate1: {predicate1}, Neighbor1: {neighbor1} ({neighbor1_label})")
            print(f"Predicate2: {predicate2}, Neighbor2: {neighbor2} ({neighbor2_label})")
            print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {e}")
