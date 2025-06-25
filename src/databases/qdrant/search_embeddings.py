from typing import List, Optional, Any, Dict

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


def fetch_similar_entities(keywords: List[Dict[str, str]], lang: str) -> Dict[str, Any]:
    items: List[Any] = []
    properties: List[Any] = []

    for k in keywords:
        keyword_type = k.get("type")
        keyword_value = k.get("value")

        if not keyword_value:
            continue

        search_results = search_wikidata(keyword=keyword_value, type=keyword_type, lang=lang)

        if keyword_type == "item":
            items.extend(search_results)
        elif keyword_type == "property":
            properties.extend(search_results)

    return {"items": items, "properties": properties}
