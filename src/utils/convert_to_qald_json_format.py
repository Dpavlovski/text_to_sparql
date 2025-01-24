import json

from src.databases.mongo.mongo import MongoDBDatabase


def convert_to_qald_mdb():
    mdb = MongoDBDatabase()
    entries = mdb.get_entries_dict("text-to-sparql")

    qald_data = {
        "dataset": {
            "id": "qald-X"
        },
        "questions": []
    }
    for id, entry in enumerate(entries):
        question_entry = {
            "id": id,
            "question": [
                {
                    "language": "en",
                    "string": entry["question"]
                }
            ],
            "answers": []
        }

        if "ask" in entry["generated_sparql"].strip().lower():
            item_result = entry["result"]
            if not isinstance(item_result, bool):
                item_result = item_result[0]
            question_entry["answers"].append({
                "head": {},
                "boolean": item_result
            })
        else:
            bindings = []

            for r in entry["result"]:
                r = str(r)
                if ":" in r:
                    li = r.split(":")[1:]
                    r = ":".join(li).strip()

                bindings.append({
                    "result": {
                        "value": r
                    }
                })

            question_entry["answers"].append({
                "head": {
                    "vars": ["result"]
                },
                "results": {
                    "bindings": bindings
                }
            })

        question_entry["query"] = {
            "sparql": entry["generated_sparql"]
        }

        qald_data["questions"].append(question_entry)

    with open('../dataset/gerbil.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(qald_data))


convert_to_qald_mdb()
