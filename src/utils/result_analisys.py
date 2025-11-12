from difflib import SequenceMatcher

import pandas as pd


def parse_ratio(ratio_str):
    """Convert a ratio like '2/5' into a float value (0.4)."""
    try:
        a, b = map(int, ratio_str.split("/"))
        return a / b if b != 0 else 0
    except Exception:
        return 0


def text_similarity(a, b):
    """Compute normalized text similarity (0–1)."""
    if not isinstance(a, str) or not isinstance(b, str):
        return 0
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def compare_result_sets(gen_results, gold_results):
    """
    Compare generated vs. gold result sets.
    Return precision, recall, and F1 for each query.
    """
    if not isinstance(gen_results, str) or not isinstance(gold_results, str):
        return 0, 0, 0

    # Split by comma and normalize
    gen_set = {x.strip() for x in gen_results.split(",") if x.strip()}
    gold_set = {x.strip() for x in gold_results.split(",") if x.strip()}

    if not gold_set and not gen_set:
        return 1, 1, 1
    if not gold_set:
        return 0, 0, 0

    intersection = len(gen_set.intersection(gold_set))
    precision = intersection / len(gen_set) if gen_set else 0
    recall = intersection / len(gold_set) if gold_set else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return precision, recall, f1


def analyze_text2sparql_dataset(file_path, similarity_threshold=0.8):
    """Analyze a Text-to-SPARQL system output CSV and measure query + result accuracy."""
    df = pd.read_csv(file_path)

    # Parse match ratio values
    df["match_ratio_value"] = df["match_ratio"].apply(parse_ratio)

    # Compute keyword ratio
    def keyword_ratio(val):
        try:
            a, b = map(int, val.split("/"))
            return a / b if b != 0 else 0
        except Exception:
            return 0

    # Compute SPARQL query similarity
    df["query_similarity"] = df.apply(
        lambda row: text_similarity(row.get("sparql", ""), row.get("gold_query", "")),
        axis=1
    )
    df["is_correct_query"] = df["query_similarity"] >= similarity_threshold

    # Compare results with gold_results
    result_metrics = df.apply(
        lambda row: compare_result_sets(row.get("results", ""), row.get("gold_results", "")),
        axis=1
    )
    df[["result_precision", "result_recall", "result_f1"]] = pd.DataFrame(result_metrics.tolist(), index=df.index)

    # Compute overall statistics
    stats = {
        "Total questions": len(df),
        "Unique questions": df["original_question"].nunique(),
        "Avg keyword F1": pd.to_numeric(df["keyword_f1"], errors="coerce").mean(),
        "Avg keyword match ratio": df["keyword_match_ratio"].apply(keyword_ratio).mean(),
        "Avg entity match precision": df["match_ratio_value"].mean(),
        "Queries with results": df["results"].notna().sum(),
        "Queries without results": df["results"].isna().sum(),
        "Success rate (queries returning results)": df["results"].notna().mean(),
        "Avg execution time (s)": df["time_of_execution"].mean(),
        "Avg result precision": df["result_precision"].mean(),
        "Avg result recall": df["result_recall"].mean(),
        "Avg result F1": df["result_f1"].mean(),
    }

    # Display as table
    stats_df = pd.DataFrame(stats.items(), columns=["Metric", "Value"])
    print("\n=== Text-to-SPARQL System Analysis ===\n")
    print(stats_df.to_string(index=False, formatters={"Value": "{:.3f}".format}))
    return df, stats_df


if __name__ == "__main__":
    file_path = "../../results/benchmark/sparql_outputs_mk_with_gold_results.csv"
    df, stats_df = analyze_text2sparql_dataset(file_path, similarity_threshold=0.8)
