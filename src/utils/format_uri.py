import re


def extract_id_from_uri(uri):
    """
    Extracts a Wikidata ID (e.g., Q42, P31) from a string or URI.
    """
    if not isinstance(uri, str):
        return None

    entity_id_match = re.search(r'([QLPSM]\d+)', uri)

    if not entity_id_match:
        return None

    return entity_id_match.group()
