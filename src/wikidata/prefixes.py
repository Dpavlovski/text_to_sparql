import re

WELL_KNOWN_PREFIXES = {
    "bd": "http://www.bigdata.com/rdf#",
    "bds": "http://www.bigdata.com/rdf/search#",
    "dct": "http://purl.org/dc/terms/",
    "geo": "http://www.opengis.net/ont/geosparql#",
    "hint": "http://www.bigdata.com/queryHints#",
    "mwapi": "https://www.mediawiki.org/ontology#API/",
    "owl": "http://www.w3.org/2002/07/owl#",
    "p": "http://www.wikidata.org/prop/",
    "pq": "http://www.wikidata.org/prop/qualifier/",
    "pqn": "http://www.wikidata.org/prop/qualifier/value-normalized/",
    "pqv": "http://www.wikidata.org/prop/qualifier/value/",
    "pr": "http://www.wikidata.org/prop/reference/",
    "prn": "http://www.wikidata.org/prop/reference/value-normalized/",
    "prov": "http://www.w3.org/ns/prov#",
    "prv": "http://www.wikidata.org/prop/reference/value/",
    "ps": "http://www.wikidata.org/prop/statement/",
    "psn": "http://www.wikidata.org/prop/statement/value-normalized/",
    "psv": "http://www.wikidata.org/prop/statement/value/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "schema": "http://schema.org/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "wd": "http://www.wikidata.org/entity/",
    "wdata": "http://www.wikidata.org/wiki/Special:EntityData/",
    "wdno": "http://www.wikidata.org/prop/novalue/",
    "wdref": "http://www.wikidata.org/reference/",
    "wds": "http://www.wikidata.org/entity/statement/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "wdtn": "http://www.wikidata.org/prop/direct-normalized/",
    "wdv": "http://www.wikidata.org/value/",
    "wikibase": "http://wikiba.se/ontology#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def ensure_prefixes(query: str) -> str:
    """
    Scans the SPARQL query for used prefixes (e.g., 'wd:', 'wdt:') 
    and prepends their definition if missing.
    """
    potential_prefixes = set(re.findall(r'\b([a-zA-Z0-9_]+):', query))

    headers = []
    for prefix in potential_prefixes:
        if prefix in WELL_KNOWN_PREFIXES:
            if f"PREFIX {prefix}:" not in query:
                headers.append(f"PREFIX {prefix}: <{WELL_KNOWN_PREFIXES[prefix]}>")

    if not headers:
        return query

    return "\n".join(headers) + "\n" + query
