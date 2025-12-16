import itertools
from typing import List, Dict, Any, Optional, Union


class ScoredPoint:
    def __init__(self, payload: dict, score: float = 1.0, id: int = 1):
        self.payload = payload
        self.score = score
        self.id = id


class QueryResponse:
    def __init__(self, points: List[ScoredPoint]):
        self.points = points


def _normalize_entity(entity: Any) -> Optional[Dict[str, Any]]:
    """
    Normalizes a single entity.
    UPDATED: Uses Duck Typing (hasattr) to handle both your Mock ScoredPoint
    and the real Qdrant ScoredPoint/Record without strict type checking.
    """
    if not entity:
        return None

    if hasattr(entity, 'payload'):
        payload = entity.payload
        # Safety check: payload might be None even if the attribute exists
        if not payload or 'id' not in payload:
            return None

        return {
            "id": payload.get('id'),
            "label": payload.get('value') or payload.get('label', 'N/A'),
            "description": payload.get('description', 'No description available')
        }

    elif isinstance(entity, dict):
        if 'id' not in entity:
            return None
        return {
            "id": entity.get('id'),
            "label": entity.get('label', 'N/A'),
            "description": entity.get('description', '')
        }

    return None


def map_candidates(
        wikidata_api_results: List[Dict[str, Any]],
        qdrant_results: Union[List[ScoredPoint], Any]
) -> List[Dict[str, Any]]:
    """
    Combines results.
    UPDATED: Automatically extracts the list from a QueryResponse object.
    """

    iterable_qdrant = qdrant_results
    if hasattr(qdrant_results, 'points'):
        iterable_qdrant = qdrant_results.points
    elif hasattr(qdrant_results, 'result'):  # Some older wrappers
        iterable_qdrant = qdrant_results.result

    unique_entities = {}

    for item in itertools.chain(iterable_qdrant, wikidata_api_results):
        normalized_entity = _normalize_entity(item)
        if normalized_entity and 'id' in normalized_entity:
            unique_entities[normalized_entity['id']] = normalized_entity

    return list(unique_entities.values())
