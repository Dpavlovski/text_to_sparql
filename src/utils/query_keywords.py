import json
import re

import pandas as pd

SPARQL_KEYWORDS = [
    'count', 'sum', 'avg', 'min', 'max',
    'distinct', 'where', 'filter',
    'order by', 'limit', 'group by',
    'union', 'optional', 'ask', 'select',
    'values', 'service', 'bind',
    'exists', 'not exists', 'minus', 'offset',
    'describe', 'construct', 'having', 'from', 'graph'
]


def load_ground_truth_map_from_qald(json_path):
    """
    Loads a QALD-style JSON file and creates a mapping from the Macedonian
    question string to the corresponding SPARQL query.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Ground truth file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from {json_path}")
        return None

    question_to_sparql_map = {}
    question_list = data.get('questions', [])
    if not question_list and isinstance(data, list):
        question_list = data

    for item in question_list:
        mk_question = ""
        for q_lang in item.get('question', []):
            if q_lang.get('language') == 'mk':
                mk_question = q_lang.get('string')
                break
        sparql_query = item.get('query', {}).get('sparql')
        if mk_question and sparql_query:
            question_to_sparql_map[mk_question] = sparql_query

    return question_to_sparql_map


def extract_sparql_keywords(query_str):
    if not isinstance(query_str, str):
        return set()
    query = re.sub(r'\s+', ' ', query_str.lower())  # normalize spaces
    found = set()
    for kw in SPARQL_KEYWORDS:
        if re.search(rf'\b{re.escape(kw)}\b', query):
            found.add(kw)
    return found


def calculate_keyword_metrics(row):
    """
    Calculates match ratio and F1 score for SPARQL keyword overlap.
    """
    gold = row['gold_keywords']
    gen = row['generated_keywords']

    if not gold:
        return "0/0", 0.0

    matches = len(gold & gen)
    precision = matches / len(gen) if gen else 0
    recall = matches / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    ratio = f"{matches}/{len(gold)}"

    return ratio, round(f1, 3)


def create_keyword_analysis_file(generated_results_path, ground_truth_path, output_path):
    """
    Merges ground truth from a JSON file with generated results from a CSV,
    analyzes SPARQL keyword matches, and saves the result to a new file.
    """
    print(f"📥 Loading generated results from '{generated_results_path}'...")
    try:
        df = pd.read_csv(generated_results_path)
    except FileNotFoundError:
        print(f"❌ Error: The file was not found at {generated_results_path}")
        return

    print("📖 Loading ground truth SPARQL queries...")
    question_to_gold_sparql = load_ground_truth_map_from_qald(ground_truth_path)
    if question_to_gold_sparql is None:
        return

    print("🔗 Merging with ground truth...")
    df['gold_query'] = df['original_question'].map(question_to_gold_sparql)
    missing_count = df['gold_query'].isna().sum()
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count}/{len(df)} questions missing gold queries.")
        df.dropna(subset=['gold_query'], inplace=True)
        print(f"Proceeding with {len(df)} valid question-query pairs.")

    print("🧠 Extracting keywords...")
    df['generated_keywords'] = df['sparql'].apply(extract_sparql_keywords)
    df['gold_keywords'] = df['gold_query'].apply(extract_sparql_keywords)

    print("📊 Calculating keyword ratios and F1 scores...")
    metrics = df.apply(calculate_keyword_metrics, axis=1)
    df[['keyword_match_ratio', 'keyword_f1']] = pd.DataFrame(metrics.tolist(), index=df.index)

    # Convert sets to comma-separated strings for readability
    df['generated_keywords'] = df['generated_keywords'].apply(lambda s: ', '.join(sorted(list(s))) if s else 'None')
    df['gold_keywords'] = df['gold_keywords'].apply(lambda s: ', '.join(sorted(list(s))) if s else 'None')

    output_columns = [
                         'original_question', 'gold_query', 'generated_keywords',
                         'gold_keywords', 'keyword_match_ratio', 'keyword_f1'
                     ] + [col for col in df.columns if col not in [
        'original_question', 'gold_query', 'generated_keywords',
        'gold_keywords', 'keyword_match_ratio', 'keyword_f1'
    ]]

    print("💾 Saving analysis to output file...")
    try:
        df[output_columns].to_csv(output_path, index=False)
        print(f"\n✅ Analysis complete! Saved to: {output_path}")
        print(
            "Columns added: ['gold_query', 'generated_keywords', 'gold_keywords', 'keyword_match_ratio', 'keyword_f1']")
    except Exception as e:
        print(f"❌ Error while saving the file: {e}")


if __name__ == '__main__':
    GENERATED_RESULTS_CSV = '../../results/benchmark/sparql_outputs_mk_with_analysis_2.csv'
    GROUND_TRUTH_JSON = '../../qald_10_with_mk.json'
    OUTPUT_CSV = '../../results/benchmark/sparql_outputs_mk_with_analysis.csv'

    create_keyword_analysis_file(GENERATED_RESULTS_CSV, GROUND_TRUTH_JSON, OUTPUT_CSV)
