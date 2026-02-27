import asyncio
import sys
import time
from typing import Any, List, Dict

import aiohttp
import requests

from src.http_client.session import get_session

USER_AGENT = "MyWikidataBot/1.0 (my-project-url.com; author)"

_last_request_time = 0
MIN_DELAY = 1.0


async def _rate_limit():
    """Asynchronously ensure a minimum delay between requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_DELAY:
        await asyncio.sleep(MIN_DELAY - elapsed)
    _last_request_time = time.time()


async def fetch_wikidata(params: dict) -> dict | None:
    """
    Asynchronously fetches data from the Wikidata API using aiohttp.

    Args:
        params: A dictionary of parameters for the API request.

    Returns:
        A dictionary with the JSON response or None on error.
    """
    session = get_session()

    url = "https://www.wikidata.org/w/api.php"
    retries = 3
    delay = 2

    for attempt in range(retries):
        await _rate_limit()
        try:
            async with session.get(
                    url,
                    params=params,
                    headers={"User-Agent": USER_AGENT},
                    timeout=15
            ) as response:

                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", delay))
                    print(f"Rate limited. Retrying after {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    delay *= 2
                    continue

                response.raise_for_status()
                return await response.json()

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"HTTP error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
            else:
                return None
        except Exception as e:
            print(f"Wikidata error: {e}")
            return None
    return None


async def search_wikidata(keyword: str, type: str, lang: str) -> list:
    """
    Asynchronously searches Wikidata for entities.
    """
    params = {
        "action": "wbsearchentities",
        "search": keyword,
        "type": type,
        "language": lang,
        "format": "json",
        "limit": 5
    }
    results = []
    wikidata_result = await fetch_wikidata(params)
    if wikidata_result and "search" in wikidata_result:
        results.extend(wikidata_result["search"])
    else:
        print(f"No results or error for {keyword}")
    return results


async def execute_sparql_query(query: str, retries: int = 3, delay: int = 5) -> Any:
    """
    Asynchronously executes a SPARQL query against the Wikidata endpoint using aiohttp.
    """
    session = get_session()

    endpoint_url = "https://query.wikidata.org/sparql"

    params = {
        "query": query,
        "format": "json"
    }

    for attempt in range(retries):
        try:
            await _rate_limit()
            async with session.get(
                    endpoint_url,
                    params=params,
                    headers={'Accept': 'application/sparql-results+json', 'User-Agent': USER_AGENT},
                    timeout=30
            ) as response:

                if response.status == 429 or response.status == 504:
                    print(f"⚠️ Query timed out or rate limited (HTTP {response.status}). "
                          f"Retrying in {delay} seconds...", file=sys.stderr)
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue

                response.raise_for_status()
                response_json = await response.json()

                if "results" in response_json and "bindings" in response_json["results"]:
                    bindings = response_json["results"]["bindings"]
                    return bindings[:10] if bindings else []

                elif "boolean" in response_json:
                    return response_json["boolean"]

                return []

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"SPARQL query error on attempt {attempt + 1}: {e}", file=sys.stderr)
            if attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
            else:
                return None
        except Exception as e:
            print(f"Wikidata error in executing query: {e}")
            return None
    return None


# print(asyncio.run(execute_sparql_query(
#     'SELECT ?person ?personLabel WHERE { wd:Q761383 wdt:P138 ?person . SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }')))

HEADERS = {
    'User-Agent': 'RatioAnalysisScript/1.0 (mailto:user@example.com)'
}
API_ENDPOINT = "https://www.wikidata.org/w/api.php"
ID_CHUNK_SIZE = 50  # API limit for wbgetentities


def get_wikidata_labels(entity_ids: List[str], language: str = 'en') -> Dict[str, str]:
    """
    Fetches labels for a list of Wikidata entity IDs (e.g., 'Q42', 'P31').

    This function automatically handles batching to respect the API's limit of 50 IDs
    per request. It is resilient to network errors and safely handles cases
    where an ID has no label for the specified language.

    Args:
        entity_ids: A list of Wikidata entity IDs as strings.
        language: The language code for the labels to retrieve (e.g., 'en', 'de', 'es').
                  Defaults to 'en' (English).

    Returns:
        A dictionary mapping each entity ID to its corresponding label. IDs that
        could not be found or had no label in the specified language are omitted.
    """
    if not entity_ids:
        return {}

    results: Dict[str, str] = {}

    # Split the list of IDs into chunks of 50
    for i in range(0, len(entity_ids), ID_CHUNK_SIZE):
        id_chunk = entity_ids[i:i + ID_CHUNK_SIZE]

        params = {
            'action': 'wbgetentities',
            'ids': '|'.join(id_chunk),
            'props': 'labels',
            'languages': language,
            'format': 'json',
        }

        try:
            response = requests.get(API_ENDPOINT, params=params, headers=HEADERS, timeout=15)
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            data = response.json()

            # Safely parse the response
            entities = data.get('entities', {})
            for entity_id, entity_data in entities.items():
                label_data = entity_data.get('labels', {}).get(language)
                if label_data:
                    results[entity_id] = label_data['value']

        except requests.exceptions.RequestException as e:
            print(f"--> API Error during request for IDs {id_chunk}: {e}")
            # Continue to the next chunk without crashing
            continue

    return results
#
#
# if __name__ == '__main__':
#     # --- Example Usage ---
#
#     # 1. A small list of mixed entity and property IDs
#     print("--- Example 1: Small list ---")
#     some_ids = ['Q42', 'P31', 'Q1', 'Q146']  # Douglas Adams, instance of, universe, cat
#     labels = get_wikidata_labels(some_ids)
#
#     if labels:
#         for entity_id, label in labels.items():
#             print(f"{entity_id}: {label}")
#     else:
#         print("Could not fetch labels for the small list.")
#
#     print("\n" + "=" * 30 + "\n")
#
#     # 2. A larger list to demonstrate automatic batching (more than 50 IDs)
#     print("--- Example 2: Larger list to test batching ---")
#     # A list of G20 countries + some other entities
#     large_id_list = [
#         'Q21', 'Q227', 'Q25', 'Q252', 'Q258', 'Q27', 'Q29', 'Q30', 'Q31', 'Q32', 'Q34', 'Q35',
#         'Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q43', 'Q16', 'Q17', 'Q142', 'Q145', 'Q148',
#         'Q155', 'Q159', 'Q183', 'Q184', 'Q211', 'Q212', 'Q213', 'Q214', 'Q215', 'Q217', 'Q218',
#         'Q219', 'Q221', 'Q222', 'Q223', 'Q224', 'Q225', 'Q227', 'Q228', 'Q229', 'Q232', 'Q233',
#         'Q235', 'Q236', 'Q237', 'Q238', 'Q241', 'Q242', 'Q244', 'Q262', 'Q265', 'Q267', 'Q28',
#         'Q77', 'Q79', 'Q83'  # And a few more to trigger the second batch
#     ]
#
#     print(f"Attempting to fetch labels for {len(large_id_list)} IDs...")
#     large_list_labels = get_wikidata_labels(large_id_list)
#
#     if large_list_labels:
#         print(f"Successfully fetched {len(large_list_labels)} labels.")
#         # Print a few examples from the result
#         count = 0
#         for entity_id, label in large_list_labels.items():
#             if count < 5:
#                 print(f"{entity_id}: {label}")
#                 count += 1
#         if len(large_list_labels) > 5:
#             print("...")
#     else:
#         print("Could not fetch labels for the large list.")
