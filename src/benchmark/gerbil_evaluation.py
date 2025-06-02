import json
from typing import Any

from src.dataset.qald_10 import load_qald_json
from src.main import text_to_sparql


def process_qald_and_save_json(benchmark_data: list[dict[str, Any]], filename: str):
    qald_data = {
        "dataset": {
            "id": "qald-embeddings"
        },
        "questions": []
    }

    for idx, data in enumerate(benchmark_data):
        question = data['question']
        generated_sparql, result = text_to_sparql(question)

        question_entry = {
            "id": str(idx),
            "question": [{
                "language": "en",
                "string": question
            }],
            "query": {
                "sparql": generated_sparql if generated_sparql else "ERROR: No query generated"
            },
            "answers": []
        }

        if not generated_sparql:
            question_entry["answers"].append({
                "error": "Failed to generate SPARQL query"
            })
        elif "ask" in generated_sparql.strip().lower():
            question_entry["answers"].append({
                "head": {},
                "boolean": bool(result) if result is not None else False
            })
        else:
            bindings = []
            if result is not None:
                for item in result:
                    item_str = str(item)
                    if ":" in item_str:
                        parts = item_str.split(":")
                        item_str = ":".join(parts[1:]).strip()

                    bindings.append({
                        "result": {
                            "value": item_str
                        }
                    })

            question_entry["answers"].append({
                "head": {"vars": ["result"]},
                "results": {"bindings": bindings}
            })

        qald_data["questions"].append(question_entry)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(qald_data, f, ensure_ascii=False, indent=4)


process_qald_and_save_json(load_qald_json(), 'results/embedding_no_zero_shot.json')
