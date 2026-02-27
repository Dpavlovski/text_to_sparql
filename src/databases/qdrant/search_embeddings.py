import asyncio
from typing import List, Any, Dict

from src.config.config import BenchmarkConfig
from src.databases.qdrant.qdrant import qdrant_db
from src.llm.embed_labels import embed_value
from src.utils.format_examples import format_qa_sparql_examples
from src.utils.map_candidates import map_candidates
from src.utils.re_ranking import rerank_candidates
from src.wikidata.api import search_wikidata


async def fetch_similar_qa_pairs(question: str, lang: str):
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
    Fetches entities via Qdrant (Semantic) and Wikidata API (Keyword).
    CRITICALLY: It uses the NER 'context' to filter the Wikidata API results.
    """
    if not keywords:
        return {}

    valid_keywords = [k for k in keywords if isinstance(k, dict) and k.get('value')]
    if not valid_keywords:
        return {}

    # 1. Prepare Qdrant Vectors (Using Value + Context)
    query_vectors = []
    search_queries = []

    for k in valid_keywords:
        search_text = f"{k.get('value', '')} {k.get('context', '')}".strip()
        search_queries.append(search_text)
        query_vectors.append(embed_value(search_text))

    # 2. Parallel Fetch
    # A. Qdrant Search (Semantic)
    qdrant_batch_task = qdrant_db.search_embeddings_batch(
        vectors=query_vectors,
        collection_name="qald_10_labels",
        score_threshold=0.6,
        top_k=5,
        filter={"lang": lang}
    )

    # B. Wikidata API Search (Keyword/Elastic)
    wikidata_tasks = [
        search_wikidata(keyword=k['value'], type=k.get('type', 'item'), lang=lang)
        for k in valid_keywords
    ]

    all_results = await asyncio.gather(qdrant_batch_task, *wikidata_tasks)

    qdrant_results_per_keyword = all_results[0]
    wikidata_results_per_keyword = all_results[1:]

    candidates_map: Dict[str, List[Dict[str, Any]]] = {}

    for i, keyword in enumerate(valid_keywords):
        q_res = qdrant_results_per_keyword[i] if i < len(qdrant_results_per_keyword) else []

        w_res_raw = wikidata_results_per_keyword[i] if i < len(wikidata_results_per_keyword) else []

        # 3. Apply Re-ranking to Wikidata Results
        w_res_filtered = rerank_candidates(search_queries[i], w_res_raw, threshold=0.85)

        # 4. Merge
        combined_list = map_candidates(w_res_filtered, q_res)

        # 5. Final Top 5 limit
        candidates_map[keyword['value']] = combined_list[:5]

    return candidates_map
