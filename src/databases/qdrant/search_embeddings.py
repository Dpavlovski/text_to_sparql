from typing import List, Optional

from qdrant_client.grpc import ScoredPoint
from qdrant_client.models import ScoredPoint  # Ensure this matches your Qdrant setup

from src.databases.qdrant.qdrant import QdrantDatabase


def extract_search_objects(
        value: str,
        collection_name: str,
        lang: Optional[str] = None
) -> List[ScoredPoint]:
    database = QdrantDatabase()

    search_filter = None
    if lang:
        search_filter = {"lang": lang}

    return database.search_embeddings_str(
        query=value,
        filter=search_filter,
        score_threshold=0.2,
        top_k=5,
        collection_name=collection_name
    )


def fetch_similar_entities(labels: List[str], lang: str) -> List[ScoredPoint]:
    similar_entities: List[ScoredPoint] = []
    for label in labels:
        similar_entities.extend(extract_search_objects(value=label, lang=lang, collection_name="qald_10_labels"))
    return similar_entities
