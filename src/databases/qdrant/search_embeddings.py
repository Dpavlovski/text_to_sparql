import asyncio
from typing import List, Any, Dict

from src.databases.qdrant.qdrant import qdrant_db
from src.llm.embed_labels import embed_value
from src.utils.format_candidates import format_candidates
from src.utils.format_examples import format_qa_sparql_examples
from src.wikidata.api import search_wikidata


async def fetch_similar_qa_pairs(question: str):
    """Fetches similar question-answer pairs to use as few-shot examples."""
    vector = embed_value(question)
    examples = await qdrant_db.search_embeddings(
        vector=vector,
        score_threshold=0.2,
        top_k=5,
        collection_name="lcquad2_0_mk"
    )
    return format_qa_sparql_examples(examples)


async def get_candidates(
        keywords: List[Any],
        lang: str
) -> Any:
    """
    Fetches, combines, and deduplicates entity candidates for a list of keywords.

    This version PRESERVES THE MAPPING between each keyword and its candidates.
    """
    # Filter for valid keywords to avoid errors and unnecessary API calls
    valid_keywords = [k for k in keywords if k.value]
    if not valid_keywords:
        return {}

    # query_vectors = []
    # for k in valid_keywords:
    #     search_text = f"{k.value}"
    #     query_vectors.append(embed_value(search_text))

    # qdrant_batch_task = qdrant_db.search_embeddings_batch(
    #     vectors=query_vectors,
    #     collection_name="qald_10_labels",
    #     score_threshold=0.7,
    #     top_k=5,
    #     filter={"lang": lang}
    # )

    wikidata_tasks = [
        search_wikidata(keyword=k.value, type=k.type, lang=lang)
        for k in valid_keywords
    ]

    all_results = await asyncio.gather(*wikidata_tasks)

    # qdrant_results_per_keyword = all_results[0]
    wikidata_results_per_keyword = all_results[0:]

    candidates_map: Dict[str, List[Dict[str, Any]]] = {}

    for i, keyword in enumerate(valid_keywords):
        # qdrant_candidates_for_keyword = qdrant_results_per_keyword[i] if i < len(qdrant_results_per_keyword) else []
        wikidata_candidates_for_keyword = wikidata_results_per_keyword[i] if i < len(
            wikidata_results_per_keyword) else []

        combined_list = format_candidates(
            [],
            wikidata_candidates_for_keyword
        )

        candidates_map[keyword.value] = combined_list

    return wikidata_results_per_keyword
