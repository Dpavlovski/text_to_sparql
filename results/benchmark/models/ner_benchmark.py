from typing import List

from tqdm import tqdm

from src.llm.llm_provider import llm_provider
from src.tools.ner import extract_entities, NERResponse

NER_MODELS_TO_BENCHMARK = [
    # "alibaba/tongyi-deepresearch-30b-a3b:free",
    # "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    # "deepseek/deepseek-r1:free",
    # "google/gemma-3-27b-it:free",
    # "google/gemma-3-4b-it:free",
    "kwaipilot/kat-coder-pro:free",
    # "meituan/longcat-flash-chat:free",
    # "meta-llama/llama-3.3-8b-instruct:free",
    # "meta-llama/llama-4-maverick:free",
    # "meta-llama/llama-4-scout:free",
    # "mistralai/mistral-small-3.1-24b-instruct:free",
    # "mistralai/mistral-small-3.2-24b-instruct:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-20b:free",
    # "qwen/qwen3-235b-a22b:free",
    # "qwen/qwen3-4b:free",
    # "z-ai/glm-4.5-air:free",
]


# --- Core Benchmark Logic ---

def find_hard_questions(results_csv_path: str, num_questions: int = 50) -> List[str]:
    """
    Identifies questions that failed to produce results in all 5 attempts.
    """
    print(f"Analyzing '{results_csv_path}' to find hard questions...")
    try:
        df = pd.read_csv(results_csv_path)
    except FileNotFoundError:
        print(f"ERROR: Results file not found at '{results_csv_path}'. Exiting.")
        return []

    df['result'] = df['result'].astype(str)
    failed_results = df[df['result'].str.strip().isin(['', 'nan', 'None', '[]'])]
    question_failure_counts = failed_results['original_question'].value_counts()
    hard_question_series = question_failure_counts[question_failure_counts >= 5]
    hard_questions = hard_question_series.index.tolist()

    print(f"Found {len(hard_questions)} questions that failed in all 5 attempts.")
    return hard_questions[:num_questions]


async def run_ner_benchmark():
    """Main function to run the NER-only benchmark and save results."""

    # 1. Setup
    original_results_csv = '../sparql_outputs_improved_linking.csv'
    benchmark_output_csv = 'ner_only_benchmark.csv'

    questions_to_test = find_hard_questions(original_results_csv)

    print(f"Selected {len(questions_to_test)} questions for the NER benchmark.")

    all_results = []

    print(
        f"\nStarting NER benchmark on {len(questions_to_test)} questions across {len(NER_MODELS_TO_BENCHMARK)} models...")

    # 2. Run benchmark for each model
    for model_identifier in NER_MODELS_TO_BENCHMARK:
        print(f"\n--- Testing Model for NER: {model_identifier} ---")

        try:
            llm_provider.get_model(model_identifier)
        except (ValueError, ImportError) as e:
            print(f"Could not initialize model '{model_identifier}': {e}. Skipping.")
            continue

        for question in tqdm(questions_to_test, desc=f"Model {model_identifier}"):
            try:
                # Directly invoke the extract_entities function with the specific model
                ner_response: NERResponse = await extract_entities(question, model_identifier)
                ner_response_str = str(ner_response)  # Convert the Pydantic object to a string for CSV

            except Exception as e:
                # This catches errors if the LLM produces invalid JSON that Pydantic can't parse
                print(f"Error processing question '{question}' with model '{model_identifier}': {e}")
                ner_response_str = f"ERROR: {e}"

            all_results.append({
                "model_name": model_identifier,
                "question": question,
                "ner_output": ner_response_str
            })

    # 3. Save results to CSV
    print(f"\nBenchmark finished. Saving results to '{benchmark_output_csv}'...")
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(benchmark_output_csv, index=False)

    # 4. Analyze and print summary
    print("\n--- NER Benchmark Summary ---")
    df_results['is_successful'] = ~df_results['ner_output'].str.startswith("ERROR", na=False)

    summary = df_results.groupby('model_name')['is_successful'].agg(['sum', 'count']).reset_index()
    summary = summary.rename(columns={'sum': 'successful_extractions', 'count': 'total_questions'})
    summary['error_count'] = summary['total_questions'] - summary['successful_extractions']
    summary['success_rate_%'] = (summary['successful_extractions'] / summary['total_questions']) * 100

    summary = summary.sort_values(by='success_rate_%', ascending=False)

    print(summary[['model_name', 'successful_extractions', 'error_count', 'success_rate_%']].to_string(index=False))


# evaluate_ner_quality.py (Corrected for AttributeError)

import pandas as pd
import re


def calculate_metrics(predicted_set, gold_set):
    """Calculates precision, recall, and F1-score for a set of keywords."""
    if not isinstance(predicted_set, set) or not isinstance(gold_set, set):
        return 0, 0, 0

    true_positives = len(predicted_set.intersection(gold_set))
    false_positives = len(predicted_set.difference(gold_set))
    false_negatives = len(gold_set.difference(predicted_set))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def parse_keyword_string_to_sets(keyword_str: str):
    """
    Parses a string containing a list of Keyword objects into sets of items and properties
    using a direct regular expression.
    """
    if not isinstance(keyword_str, str) or keyword_str.startswith("ERROR"):
        return set(), set()

    try:
        # This regex looks for the pattern "Keyword(value='...', type='...')"
        # and captures the content inside the single quotes for both value and type.
        pattern = re.compile(r"Keyword\(value='([^']*)',\s*type='([^']*)'\)")

        # findall will return a list of tuples, e.g., [('animal', 'item'), ('military operation', 'property')]
        matches = pattern.findall(keyword_str)

        items = set()
        props = set()

        for value, type in matches:
            if type == 'item':
                items.add(value)
            elif type == 'property':
                props.add(value)

        return items, props

    except Exception:
        # A general catch-all in case of unexpected regex errors
        return set(), set()

    except (ValueError, SyntaxError, AttributeError):
        return set(), set()


def evaluate_ner_quality():
    """
    Loads NER benchmark results and the new single-column ground truth file,
    then calculates quality metrics.
    """
    try:
        ground_truth_df = pd.read_csv('ground_truth_ner.csv')
        benchmark_df = pd.read_csv('ner_only_benchmark.csv')
    except FileNotFoundError as e:
        print(f"Error: Make sure both 'ground_truth_ner.csv' and 'ner_only_benchmark.csv' exist. Details: {e}")
        return

    df = pd.merge(benchmark_df, ground_truth_df, on='question')
    results = []

    for _, row in df.iterrows():
        # --- Parse BOTH columns using the SAME robust parser ---
        gold_items, gold_props = parse_keyword_string_to_sets(row['gold_keywords'])
        predicted_items, predicted_props = parse_keyword_string_to_sets(row['ner_output'])

        # Calculate metrics for items and properties separately
        item_p, item_r, item_f1 = calculate_metrics(predicted_items, gold_items)
        prop_p, prop_r, prop_f1 = calculate_metrics(predicted_props, gold_props)

        # Calculate overall F1 score
        overall_predicted = predicted_items.union(predicted_props)
        overall_gold = gold_items.union(gold_props)
        _, _, overall_f1 = calculate_metrics(overall_predicted, overall_gold)

        results.append({
            'model_name': row['model_name'],
            'question': row['question'],
            'item_f1': item_f1,
            'property_f1': prop_f1,
            'overall_f1': overall_f1,
        })

    if not results:
        print("No valid results to analyze. Check parsing logic and input files.")
        return

    results_df = pd.DataFrame(results)
    summary = results_df.groupby('model_name')[['item_f1', 'property_f1', 'overall_f1']].mean().reset_index()
    summary = summary.sort_values(by='overall_f1', ascending=False)

    print("\n--- NER Quality Benchmark Summary (Average F1-Score) ---")
    print(summary.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    # asyncio.run(run_ner_benchmark())
    evaluate_ner_quality()
