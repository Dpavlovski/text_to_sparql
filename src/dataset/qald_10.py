import json
from typing import Any


def load_qald_json() -> list[dict[str, Any]]:
    with open("C:\\Users\\User\\PycharmProjects\\text_to_sparql\\src\\dataset\\qald_10.json", "r",
              encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    rows = []
    for question in questions:
        q = question['question'][0]['string']
        ground_truth_sparql = question['query']['sparql']

        if 'boolean' in question['answers'][0]:
            expected_result = question['answers'][0]['boolean']
        elif 'results' in question['answers'][0]:
            results = question['answers'][0]['results']
            if 'bindings' in results and results['bindings']:
                expected_result = [
                    binding['result']['value']
                    for binding in results['bindings']
                    if 'result' in binding and 'value' in binding['result']
                ]
            else:
                expected_result = "No results found."
        else:
            expected_result = "No results found."

        row = {
            "question": q,
            "ground_truth_sparql": ground_truth_sparql,
            "expected_result": expected_result,
        }
        rows.append(row)

    return rows
