import json
from typing import Any

from src.dataset.qald_10 import load_qald_json
from src.main import text_to_sparql


def evaluation(benchmark_data: list[dict[str, Any]], file_name: str):
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
                "sparql": generated_sparql
            },
            "answers": []
        }

        if "ask" in generated_sparql.strip().lower():
            answer_entry = {
                "head": {},
                "boolean": bool(result)
            }
        else:
            bindings = []
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

            answer_entry = {
                "head": {"vars": ["result"]},
                "results": {"bindings": bindings}
            }

        question_entry["answers"].append(answer_entry)
        qald_data["questions"].append(question_entry)

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(qald_data, f, ensure_ascii=False, indent=4)


evaluation(load_qald_json(), 'embedding_all_steps.json')
