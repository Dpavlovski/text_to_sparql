import asyncio
from typing import List, Optional, Any, Dict

from qdrant_client.http.models import ScoredPoint

from src.databases.qdrant.qdrant import qdrant_db
from src.wikidata.api import search_wikidata


async def search_embeddings(
        value: str,
        collection_name: str,
        lang: Optional[str] = None
) -> List[ScoredPoint]:
    search_filter = {"lang": lang} if lang else None
    return await qdrant_db.search_embeddings_str(
        query=value,
        filter=search_filter,
        score_threshold=0.2,
        top_k=5,
        collection_name=collection_name
    )


async def fetch_similar_entities(keywords: List[Any], lang: str) -> Dict[str, Any]:
    items: List[Any] = []
    properties: List[Any] = []

    valid_keywords = [k for k in keywords if k.value]

    # --- Step 1: Create all concurrent tasks ---

    # Create a list of tasks for embedding searches
    embedding_tasks = [
        # search_embeddings(
        #     k.value,
        #     collection_name="wikidata_labels_en",
        #     lang=lang
        # ) for k in valid_keywords
    ]

    # Create a list of tasks for Wikidata API searches
    wikidata_tasks = [
        search_wikidata(
            keyword=k.value,
            type=k.type,
            lang=lang
        ) for k in valid_keywords
    ]

    # --- Step 2: Run all tasks concurrently ---

    # Run all tasks and wait for all of them to complete.
    # The `*` unpacks the lists into individual arguments for gather.
    results = await asyncio.gather(
        *embedding_tasks,
        *wikidata_tasks
    )

    # --- Step 3: Process the results ---

    # The first N results belong to the embedding tasks
    embedding_results = results[:len(embedding_tasks)]
    final_embeddings = [emb for result in embedding_results for emb in result]

    # The remaining results belong to the wikidata tasks
    wikidata_results = results[len(embedding_tasks):]

    for i, k in enumerate(valid_keywords):
        search_results = wikidata_results[i]
        if search_results:
            if k.type == "item":
                items.extend(search_results)
            elif k.type == "property":
                properties.extend(search_results)

    return {"items": items, "properties": properties, "embeddings": final_embeddings}
