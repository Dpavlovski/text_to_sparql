import logging
import re


def extract_id_from_uri(uri):
    entity_id_match = re.search(r'([QLPSM]\d+)', uri)
    if not entity_id_match:
        logging.warning(f"Invalid entity URI: {uri}")
        return None

    return entity_id_match.group()
