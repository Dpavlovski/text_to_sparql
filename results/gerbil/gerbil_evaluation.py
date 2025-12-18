import csv
import json


def gerbil_eval(csv_filename: str, json_filename: str):
    qald_data = {
        "dataset": {"id": "qald-X"},
        "questions": []
    }

    null_values = {'null', 'N/A', ''}

    attempts = []  # holds (idx, row) for the current question group
    current_question = None

    def process_attempts(attempts_group):
        """
        Decide which attempt to keep for a group.
        """
        # Find first attempt in the group with non-empty results
        for idx, row in attempts_group:
            if row.get('result') not in null_values:
                return idx, row  # keep this attempt
        # Otherwise keep the last attempt in the group
        return attempts_group[-1]

    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        idx = 0
        for row in reader:
            qtext = row.get('original_question', None)

            if current_question is None:
                current_question = qtext

            # When question changes or we hit 5 attempts, process the group
            if qtext != current_question or len(attempts) == 5:
                if attempts:
                    keep_idx, keep_row = process_attempts(attempts)
                    # save only that attempt
                    question_entry = transform_entry(keep_idx, keep_row, null_values)
                    idx += 1

                    qald_data["questions"].append(question_entry)
                # reset group
                attempts = []
                current_question = qtext

            attempts.append((idx, row))

        # Process leftover attempts at EOF
        if attempts:
            keep_idx, keep_row = process_attempts(attempts)
            qald_data["questions"].append(transform_entry(keep_idx, keep_row, null_values))

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(qald_data, f, ensure_ascii=False, indent=4)


def transform_entry(idx: int, row: dict, null_values: set):
    question = row.get('original_question', None)
    sparql = row.get('gold_query', None)
    results = row.get('gold_result', None)

    entry = {
        "id": idx,
        "question": [{"language": "en", "string": question}],
        "answers": [],
        "query": {"sparql": sparql if sparql else None}
    }

    if not sparql:
        entry["answers"].append({"sparql": None})
    elif "ask" in sparql.strip().lower():
        entry["answers"].append({
            "head": {},
            "boolean": bool(results) if results is not None else False
        })
    else:
        answer = {"head": {"vars": ["result"]}}
        if results not in null_values:
            bindings = []
            for item in results.split("\n"):
                bindings.append({
                    "result": {
                        "type": "uri" if item.startswith("http") else "literal",
                        "value": item
                    }
                })
            answer["results"] = {"bindings": bindings}
        entry["answers"].append(answer)

    return entry


if __name__ == '__main__':
    gerbil_eval('../benchmark/with_neighbors/sparql_outputs_en_gpt-4.1-mini.csv', 'test_gold.json')
