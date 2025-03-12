from typing import List

from qdrant_client.grpc import ScoredPoint

from src.databases.qdrant.qdrant import QdrantDatabase


def extract_search_objects(value: str, lang: str, collection_name: str) -> List[ScoredPoint]:
    database = QdrantDatabase()
    return database.search_embeddings_str(
        query=value,
        filter={'lang': lang},
        score_threshold=0.2,
        top_k=5,
        collection_name=collection_name
    )


def fetch_similar_entities(labels: List[str], lang: str) -> List[ScoredPoint]:
    similar_entities: List[ScoredPoint] = []
    for label in labels:
        similar_entities.extend(extract_search_objects(value=label, lang=lang, collection_name="qald_10_labels"))
    return similar_entities
