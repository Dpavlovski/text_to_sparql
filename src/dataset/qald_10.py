import json
from typing import Any


def load_qald_json() -> list[dict[str, Any]]:
    with open("../dataset/qald_10.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    rows = []
    for question in questions:
        q = question['question'][0]['string']
        ground_truth_sparql = question['query']['sparql']
        expected_result = question['answers'][0].get('results', "No results found.")
        if expected_result != "No results found.":
            expected_result = expected_result['bindings'][0]['result']['value']
        row = {
            "question": q,
            "ground_truth_sparql": ground_truth_sparql,
            "expected_result": expected_result,
        }
        rows.append(row)

    return rows

# print(load_qald_json())
