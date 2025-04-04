from typing import Any

from src.databases.mongo.mongo import MongoEntry


class EmbeddingQALD10(MongoEntry):
    question: str
    ground_truth_sparql: str
    expected_result: Any
    generated_sparql: Any
    result: Any
