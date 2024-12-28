import json

import requests


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
