from typing import List, Dict, Any

import numpy as np

from src.llm.embed_labels import embedder, embed_value


def rerank_candidates(
        target_query: str,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.70
) -> List[Dict[str, Any]]:
    """
    Takes a list of candidates (e.g. from Wikidata API), embeds their
    Label+Description, and keeps only those similar to target_query.
    """
    if not candidates or not target_query:
        return candidates

    # 1. Embed the User's Intent (Keyword + Context)
    target_vec = embed_value(target_query)

    # 2. Prepare Text for Candidates
    cand_texts = [
        f"{c.get('label', '')} {c.get('description', '') or ''}"
        for c in candidates
    ]

    # 3. Batch Embed Candidates
    cand_vecs = embedder.embed_batch(cand_texts)

    # 4. Calculate Cosine Similarity
    # (Simple numpy implementation)
    target_arr = np.array(target_vec)
    cand_arr = np.array(cand_vecs)

    norm_target = np.linalg.norm(target_arr)
    norm_cands = np.linalg.norm(cand_arr, axis=1)

    # Dot product / (norm * norm)
    scores = np.dot(cand_arr, target_arr) / (norm_cands * norm_target + 1e-10)

    # 5. Filter
    filtered = []
    for i, score in enumerate(scores):
        if score >= threshold:
            candidates[i]['_score'] = float(score)  # Save score for debugging
            filtered.append(candidates[i])

    # Fallback: if we filtered everything, keep top 1 just in case
    if not filtered and candidates:
        top_idx = np.argmax(scores)
        return [candidates[top_idx]]

    # Sort best match first
    filtered.sort(key=lambda x: x['_score'], reverse=True)
    return filtered
