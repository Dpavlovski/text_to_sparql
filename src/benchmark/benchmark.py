from typing import Any

from src.databases.mongo.mongo import MongoDBDatabase
from src.dataset.qald_10 import load_qald_json
from src.main import text_to_sparql


def test_on_qald_10(benchmark_data: list[dict[str, Any]]):
    db = MongoDBDatabase()
    for data in benchmark_data[0:30]:
        question = data['question']
        ground_truth_sparql = data['ground_truth_sparql']
        expected_result = data['expected_result']

        generated_sparql, result = text_to_sparql(question)
        entry = {
            "question": question,
            "ground_truth_sparql": ground_truth_sparql,
            "expected_result": expected_result,
            "generated_sparql": generated_sparql,
            "result": result,
        }
        db.add_entry_dict(entity=entry, collection_name="text-to-sparql")


test_on_qald_10(load_qald_json())
