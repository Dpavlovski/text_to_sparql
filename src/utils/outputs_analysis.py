import ast
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Set, Tuple, Dict, Any

import pandas as pd

# Import the label fetcher from your existing API module
from src.wikidata.api import get_wikidata_labels


class SPARQLUtils:
    """Helper methods for string extraction and normalization."""

    SPARQL_KEYWORDS = {
        'count', 'sum', 'avg', 'min', 'max', 'distinct', 'where', 'filter',
        'order by', 'limit', 'group by', 'union', 'optional', 'ask', 'select',
        'values', 'service', 'bind', 'exists', 'not exists', 'minus', 'offset',
        'describe', 'construct', 'having', 'from', 'graph'
    }

    @staticmethod
    def extract_ids_from_text(text: str) -> Set[str]:
        """Extracts all unique Wikidata IDs (Q.../P...) from a SPARQL string."""
        if not isinstance(text, str):
            return set()
        return set(re.findall(r'\b([QP]\d+)\b', text))

    @staticmethod
    def extract_keywords(query: str) -> Set[str]:
        """Extracts SPARQL keywords from a query string."""
        if not isinstance(query, str):
            return set()
        query_lower = re.sub(r'\s+', ' ', query.lower())
        found = set()
        for kw in SPARQLUtils.SPARQL_KEYWORDS:
            if re.search(rf'\b{re.escape(kw)}\b', query_lower):
                found.add(kw)
        return found

    @staticmethod
    def normalize_result_string(result_str: Any) -> Set[str]:
        """Converts results (JSON list or comma string) to a set for comparison."""
        if not result_str or pd.isna(result_str):
            return set()

        str_val = str(result_str).strip()

        if str_val.startswith('[') and str_val.endswith(']'):
            try:
                val_list = ast.literal_eval(str_val)
                return {str(x).strip() for x in val_list}
            except:
                pass

        return {x.strip() for x in str_val.split(',') if x.strip()}

    @staticmethod
    def calculate_f1(generated_set: Set[str], gold_set: Set[str]) -> Tuple[float, float, float]:
        """Returns (Precision, Recall, F1)."""
        if not generated_set and not gold_set:
            return 1.0, 1.0, 1.0
        if not gold_set:
            return 0.0, 0.0, 0.0

        tp = len(generated_set.intersection(gold_set))
        precision = tp / len(generated_set) if generated_set else 0.0
        recall = tp / len(gold_set)

        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    @staticmethod
    def text_similarity(a: str, b: str) -> float:
        """Compute normalized text similarity (0–1)."""
        if not isinstance(a, str) or not isinstance(b, str):
            return 0.0
        return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def load_qald_json(json_path: str, lang: str = 'en') -> Dict[str, Dict[str, Any]]:
    """Loads QALD JSON and maps questions to Gold SPARQL/Results."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return {}

    gold_map = {}
    questions_list = data.get('questions', data) if isinstance(data, dict) else data

    for entry in questions_list:
        q_id = str(entry.get('id', ''))
        q_text = None
        for q in entry.get('question', []):
            if q.get('language') == lang:
                q_text = q.get('string', '').strip()
                break

        if not q_text: continue

        sparql = entry.get('query', {}).get('sparql', '')
        answers_list = []
        for ans in entry.get('answers', []):
            if 'results' in ans:
                for binding in ans['results'].get('bindings', []):
                    for var_val in binding.values():
                        answers_list.append(var_val.get('value', ''))
            elif 'boolean' in ans:
                answers_list.append(str(ans['boolean']).lower())

        gold_map[q_text] = {
            "id": q_id,
            "gold_query": sparql,
            "gold_result": ", ".join(answers_list)
        }

    return gold_map


class AnalysisPipeline:
    def __init__(self, generated_csv: str, qald_json: str, lang: str = 'en'):
        self.gen_path = Path(generated_csv)
        self.json_path = Path(qald_json)
        self.lang = lang
        self.df = pd.DataFrame()
        self.label_cache = {}  # Cache to store fetched labels

    def fetch_all_labels(self, all_ids: Set[str]):
        """Fetches labels for all unique IDs found in the dataset."""
        print(f"🌍 Fetching labels for {len(all_ids)} unique entities...")
        if not all_ids:
            return

        # Convert to list for the API function
        ids_list = list(all_ids)

        try:
            # Assumes get_wikidata_labels handles chunking (50 IDs limit) internally
            # If your get_wikidata_labels is strictly for list->dict, this works directly.
            labels = get_wikidata_labels(ids_list, language=self.lang)
            self.label_cache.update(labels)
            print(f"✅ Cached {len(self.label_cache)} labels.")
        except Exception as e:
            print(f"⚠️ Warning: Failed to fetch labels: {e}")

    def format_ids(self, ids_set: Set[str]) -> str:
        """Formats a set of IDs into 'Label (ID)' string."""
        formatted = []
        for qid in sorted(list(ids_set)):
            label = self.label_cache.get(qid, "Unknown")
            formatted.append(f"{label} ({qid})")
        return ", ".join(formatted)

    def run(self, output_path: str):
        print(f"🚀 Starting Analysis...")

        # 1. Load Data
        self.df = pd.read_csv(self.gen_path)
        qald_map = load_qald_json(self.json_path, self.lang)
        print(f"ℹ️  Loaded {len(qald_map)} QALD entries.")

        # 2. Merge Data
        self.df['original_question_clean'] = self.df['original_question'].astype(str).str.strip()

        def enrich_row(row):
            q_text = row['original_question_clean']
            qald_data = qald_map.get(q_text, {})
            return pd.Series([
                qald_data.get('id', ''),
                qald_data.get('gold_query', ''),
                qald_data.get('gold_result', '')
            ])

        print("🔗 Merging datasets...")
        self.df[['question_id', 'gold_query', 'gold_result']] = self.df.apply(enrich_row, axis=1)

        # --- PRE-FETCH LABELS ---
        print("🔍 Extracting all IDs for label lookup...")
        all_unique_ids = set()
        for _, row in self.df.iterrows():
            gen_q = str(row.get('generated_query', row.get('sparql', '')))
            gold_q = str(row.get('gold_query', ''))
            all_unique_ids.update(SPARQLUtils.extract_ids_from_text(gen_q))
            all_unique_ids.update(SPARQLUtils.extract_ids_from_text(gold_q))

        self.fetch_all_labels(all_unique_ids)

        # 3. Calculate ID Metrics (With Label Formatting)
        def calc_id_metrics(row):
            gen_query = str(row.get('generated_query', row.get('sparql', '')))
            gold_query = str(row.get('gold_query', ''))
            gen_ids = SPARQLUtils.extract_ids_from_text(gen_query)
            gold_ids = SPARQLUtils.extract_ids_from_text(gold_query)

            # Use formatted strings with labels
            gen_ids_list = self.format_ids(gen_ids)
            gold_ids_list = self.format_ids(gold_ids)

            if not gold_ids: return gen_ids_list, gold_ids_list, 0.0
            matches = len(gen_ids.intersection(gold_ids))
            return gen_ids_list, gold_ids_list, (matches / len(gold_ids))

        self.df[['candidate_ids', 'gold_wikidata_ids', 'id_match_score']] = self.df.apply(
            lambda r: pd.Series(calc_id_metrics(r)), axis=1
        )

        # 4. Calculate Keyword Metrics
        def calc_kw_metrics(row):
            gen_query = str(row.get('generated_query', row.get('sparql', '')))
            gold_query = str(row.get('gold_query', ''))
            gen_set = SPARQLUtils.extract_keywords(gen_query)
            gold_set = SPARQLUtils.extract_keywords(gold_query)

            gen_list_str = str(sorted(list(gen_set)))
            gold_list_str = str(sorted(list(gold_set)))
            _, _, f1 = SPARQLUtils.calculate_f1(gen_set, gold_set)
            return gen_list_str, gold_list_str, round(f1, 3)

        self.df[['generated_keywords', 'gold_keywords', 'keyword_match_ratio']] = self.df.apply(
            lambda r: pd.Series(calc_kw_metrics(r)), axis=1
        )

        # 5. Calculate Result Metrics
        def calc_res(row):
            gen = SPARQLUtils.normalize_result_string(row.get('result'))
            gold = SPARQLUtils.normalize_result_string(row.get('gold_result'))
            p, r, f1 = SPARQLUtils.calculate_f1(gen, gold)
            return round(p, 3), round(r, 3), round(f1, 3)

        self.df[['res_precision', 'res_recall', 'res_f1']] = self.df.apply(
            lambda r: pd.Series(calc_res(r)), axis=1
        )

        self.df['query_similarity'] = self.df.apply(
            lambda row: SPARQLUtils.text_similarity(
                str(row.get('generated_query', row.get('sparql', ''))),
                str(row.get('gold_query', ''))
            ), axis=1
        )

        self._print_detailed_analysis()

        if 'original_question_clean' in self.df.columns:
            del self.df['original_question_clean']

        # 6. Save Final Columns
        final_cols = [
            'original_question',
            'generated_query', 'gold_query',
            'candidate_ids', 'gold_wikidata_ids', 'id_match_score',
            'generated_keywords', 'gold_keywords', 'keyword_match_ratio',
            'result', 'gold_result', 'res_f1'
        ]

        # Keep extra columns but exclude intermediate metrics unless needed
        existing_extra = [c for c in self.df.columns if c not in final_cols and c not in [
            'question_id', 'res_precision', 'res_recall', 'query_similarity'
        ]]

        self.df = self.df[final_cols + existing_extra]
        self.df.to_csv(output_path, index=False)
        print(f"\n💾 Analysis saved to: {output_path}")

    def _print_detailed_analysis(self):
        """Prints the analysis table."""
        valid_df = self.df[self.df['gold_query'] != '']

        def has_results(val):
            return pd.notna(val) and str(val).strip() != ""

        queries_with_results = valid_df['result'].apply(has_results).sum()
        queries_without_results = len(valid_df) - queries_with_results

        stats = {
            "Total questions": len(valid_df),
            "Unique questions": valid_df['original_question'].nunique(),

            "Avg Entity Match Score": valid_df['id_match_score'].mean(),
            "Avg Keyword Match Ratio": valid_df['keyword_match_ratio'].mean(),

            "Avg SPARQL Text Similarity": valid_df['query_similarity'].mean(),

            "Queries with results": queries_with_results,
            "Queries without results": queries_without_results,
            "Success rate (queries returning results)": queries_with_results / len(valid_df) if len(
                valid_df) > 0 else 0,

            "Avg Result Precision": valid_df['res_precision'].mean(),
            "Avg Result Recall": valid_df['res_recall'].mean(),
            "Avg Result F1": valid_df['res_f1'].mean(),
        }

        stats_df = pd.DataFrame(stats.items(), columns=["Metric", "Value"])

        print("\n" + "=" * 50)
        print("=== Text-to-SPARQL System Analysis ===")
        print("=" * 50)
        print(stats_df.to_string(index=False, formatters={"Value": "{:.3f}".format}))
        print("=" * 50 + "\n")


if __name__ == "__main__":
    GENERATED_CSV = "../../results/benchmark/with_neighbors/sparql_outputs_ru_nemotron-3-nano-30b-a3.csv"
    QALD_JSON = "../../qald_10_with_mk.json"
    OUTPUT_CSV = "../../results/benchmark/with_neighbors/sparql_outputs_ru_nemotron-3-nano-30b-a3_with_analysis.csv"
    LANGUAGE = "ru"

    if Path(GENERATED_CSV).exists() and Path(QALD_JSON).exists():
        pipeline = AnalysisPipeline(GENERATED_CSV, QALD_JSON, LANGUAGE)
        pipeline.run(OUTPUT_CSV)
    else:
        print("❌ Error: Input files not found. Check paths.")