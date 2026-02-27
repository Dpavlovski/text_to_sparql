import asyncio
from typing import List

from src.wikidata.api import execute_sparql_query

OUTGOING_QUERY = """
SELECT DISTINCT ?propLabel ?valLabel WHERE {{
  wd:{qid} ?p ?val .
  ?prop wikibase:directClaim ?p .
  ?prop rdfs:label ?propLabel .
  ?val rdfs:label ?valLabel .
  FILTER(LANG(?propLabel) = "en" && LANG(?valLabel) = "en")
}} LIMIT 5
"""

INCOMING_QUERY = """
SELECT DISTINCT ?subjLabel ?propLabel WHERE {{
  ?subj ?p wd:{qid} .
  ?prop wikibase:directClaim ?p .
  ?prop rdfs:label ?propLabel .
  ?subj rdfs:label ?subjLabel .
  FILTER(LANG(?propLabel) = "en" && LANG(?subjLabel) = "en")
}} LIMIT 5
"""


async def get_entity_neighbors(qid: str) -> List[str]:
    """
    Fetches a few examples of how this entity is connected in the graph.
    Returns a list of strings like "Riemannian geometry -> discoverer or inventor -> Bernhard Riemann"
    """
    if not qid.startswith("Q"):
        return []

    results = []

    try:
        out_res, in_res = await asyncio.gather(
            execute_sparql_query(OUTGOING_QUERY.format(qid=qid)),
            execute_sparql_query(INCOMING_QUERY.format(qid=qid))
        )

        if out_res:
            for r in out_res:
                p = r.get("propLabel", {}).get("value", "?")
                v = r.get("valLabel", {}).get("value", "?")
                results.append(f"  - (This) -> [{p}] -> {v}")

        if in_res:
            for r in in_res:
                s = r.get("subjLabel", {}).get("value", "?")
                p = r.get("propLabel", {}).get("value", "?")
                results.append(f"  - {s} -> [{p}] -> (This)")

    except Exception as e:
        print(f"Error fetching neighbors for {qid}: {e}")

    return results


async def enrich_candidates(candidates_map):
    enrichment_tasks = []

    enrichment_targets = []

    for mention, cand_list in candidates_map.items():
        if cand_list:
            for cand in cand_list:
                qid = cand.get('id')
                if qid and str(qid).startswith('Q'):
                    enrichment_tasks.append(get_entity_neighbors(qid))
                    enrichment_targets.append(cand)

    if enrichment_tasks:
        results = await asyncio.gather(*enrichment_tasks)
        for cand, neighbors in zip(enrichment_targets, results):
            cand['neighbors'] = neighbors
