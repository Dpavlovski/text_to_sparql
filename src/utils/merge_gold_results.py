import json

import pandas as pd


def merge_gold_results(csv_path, json_path, output_path):
    """
    Merge gold results from a QALD-10 JSON file into an existing Text-to-SPARQL CSV.
    Adds a 'gold_results' column matching by the question text.
    Handles both SELECT (results/bindings) and ASK (boolean) query types.
    Outputs plain comma-separated strings (not Python list syntax).
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        qald_data = json.load(f)

    gold_map = {}

    for entry in qald_data.get("questions", []):
        question_text = None
        gold_answers = []

        # Get question text (prefer Macedonian)
        for q in entry.get("question", []):
            if q.get("language") == "mk":
                question_text = q.get("string", "").strip()
                break

        if not question_text:
            continue

        # Extract answers (SELECT or ASK)
        answers = entry.get("answers", [])
        for a in answers:
            if "results" in a:
                bindings = a["results"].get("bindings", [])
                for binding in bindings:
                    for v in binding.values():
                        gold_answers.append(v.get("value", ""))
            elif "boolean" in a:
                gold_answers.append(str(a["boolean"]))

        # Convert list to clean comma-separated string
        gold_map[question_text] = ", ".join(gold_answers) if gold_answers else ""

    # Append gold results to CSV
    df["gold_results"] = df["original_question"].apply(lambda q: gold_map.get(q.strip(), ""))

    # Save updated dataset
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Merged dataset saved to: {output_path}")
    print(f"Total questions matched: {(df['gold_results'] != '').sum()} / {len(df)}")


if __name__ == "__main__":
    merge_gold_results(
        csv_path="../../results/benchmark/sparql_outputs_mk_with_analysis.csv",
        json_path="../../qald_10_with_mk.json",
        output_path="../../results/benchmark/sparql_outputs_mk_with_gold_results.csv"
    )
