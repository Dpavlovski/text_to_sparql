import itertools
from typing import List, Dict, Any, Optional


# --- Mocked Dependencies for a runnable example ---
# In your project, you would use your actual imports.


class ScoredPoint:
    def __init__(self, payload: dict, score: float = 1.0, id: int = 1):
        self.payload = payload
        self.score = score
        self.id = id

    def __repr__(self):
        return f"ScoredPoint(payload={self.payload})"


# --- Utility Functions for Candidate Processing ---

def _normalize_entity(entity: Any) -> Optional[Dict[str, Any]]:
    """
    Normalizes a single entity from any source (Qdrant or dict) into a standard dictionary format.
    """
    if not entity:
        return None
    if isinstance(entity, ScoredPoint):
        payload = entity.payload
        if not payload or 'id' not in payload: return None
        return {
            "id": payload.get('id'),
            "label": payload.get('value') or payload.get('label', 'N/A'),
            "description": payload.get('description', 'No description available')
        }
    elif isinstance(entity, dict):
        if 'id' not in entity: return None
        return {
            "id": entity.get('id'),
            "label": entity.get('label', 'N/A'),
            "description": entity.get('description', 'No description available')
        }
    return None


def format_candidates(
        qdrant_results: List[ScoredPoint],
        wikidata_api_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combines, normalizes, and deduplicates candidates from two sources based on entity 'id'.
    (This function was named format_candidates in your code, renaming for clarity is optional).
    """
    unique_entities = {}
    # Use itertools.chain to efficiently iterate over both lists at once
    for item in itertools.chain(qdrant_results, wikidata_api_results):
        normalized_entity = _normalize_entity(item)
        if normalized_entity and 'id' in normalized_entity:
            unique_entities[normalized_entity['id']] = normalized_entity
    return list(unique_entities.values())
