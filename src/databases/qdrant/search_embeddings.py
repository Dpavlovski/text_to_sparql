from src.databases.qdrant.qdrant import QdrantDatabase


async def fetch_similar_qa_pairs(question: str, lang: str, qdrant_db: QdrantDatabase):
    """Fetches similar question-answer pairs using language-specific collection."""
    config = BenchmarkConfig(lang)
    collection_name = config.get_collection_name("few_shot")
    vector = embed_value(question)

    if not await qdrant_db.collection_exists(collection_name):
        return ""

    examples = await qdrant_db.search_embeddings(
        vector=vector,
        score_threshold=0.2,
        top_k=5,
        collection_name=collection_name
    )
    return format_qa_sparql_examples(examples)


import asyncio
from typing import List, Any, Dict

from src.config.config import BenchmarkConfig
from src.databases.qdrant.embed_labels import embed_value
from src.databases.qdrant.qdrant import QdrantDatabase
from src.utils.format_examples import format_qa_sparql_examples
from src.utils.re_ranking import rerank_candidates
from src.wikidata.api import search_wikidata


async def fetch_similar_qa_pairs(question: str, lang: str, qdrant_db: QdrantDatabase):
    """Fetches similar question-answer pairs using language-specific collection."""
    config = BenchmarkConfig(lang)
    collection_name = config.get_collection_name("few_shot")
    vector = embed_value(question)

    if not await qdrant_db.collection_exists(collection_name):
        return ""

    examples = await qdrant_db.search_embeddings(
        vector=vector,
        score_threshold=0.2,
        top_k=5,
        collection_name=collection_name
    )
    return format_qa_sparql_examples(examples)


async def get_candidates(
        keywords: List[Dict[str, Any]],
        lang: str
) -> Any:
    """
    Fetches entities via Wikidata API (Keyword/ElasticSearch) ONLY.
    Applies semantic reranking locally to disambiguate and resolve homonyms.
    """
    if not keywords:
        return {}

    valid_keywords = [k for k in keywords if isinstance(k, dict) and k.get('value')]
    if not valid_keywords:
        return {}

    search_queries = []

    # 1. Prepare search text (Value + Context from NER)
    for k in valid_keywords:
        search_text = f"{k.get('value', '')} {k.get('context', '')}".strip()
        search_queries.append(search_text)

    # 2. Parallel Fetch ONLY from Wikidata API (ElasticSearch)
    wikidata_tasks = [
        search_wikidata(keyword=k['value'], type=k.get('type', 'item'), lang=lang)
        for k in valid_keywords
    ]

    wikidata_results_per_keyword = await asyncio.gather(*wikidata_tasks)

    candidates_map: Dict[str, List[Dict[str, Any]]] = {}

    # 3. Apply Local Semantic Re-ranking
    for i, keyword in enumerate(valid_keywords):
        w_res_raw = wikidata_results_per_keyword[i] if i < len(wikidata_results_per_keyword) else []

        # This uses the local HuggingFace model in memory to do Cosine Similarity!
        w_res_filtered = rerank_candidates(search_queries[i], w_res_raw, threshold=0.85)

        # 4. Format the final cleaned list
        final_list = []
        for item in w_res_filtered:
            final_list.append({
                "id": item.get('id'),
                "label": item.get('label', 'N/A'),
                "description": item.get('description', '')
            })

        # Keep top 5 after reranking
        candidates_map[keyword['value']] = final_list[:5]

    return candidates_map
