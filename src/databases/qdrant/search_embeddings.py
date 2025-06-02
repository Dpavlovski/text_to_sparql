from typing import List, Optional, Any

from qdrant_client.http.models import ScoredPoint

from src.databases.qdrant.qdrant import QdrantDatabase
from src.wikidata.api import search_wikidata


def search_embeddings(
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


def fetch_similar_entities(labels: List[str], lang: str) -> List[Any]:
    similar_entities: List[ScoredPoint] = []
    for label in labels:
        similar_entities.extend(search_wikidata(keyword=label, lang=lang))
        similar_entities.extend(search_embeddings(value=label, lang=lang, collection_name="qald_10_labels"))
    return similar_entities
