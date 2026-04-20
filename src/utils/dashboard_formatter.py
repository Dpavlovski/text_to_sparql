# src/utils/dashboard_formatter.py
from pathlib import Path

import pandas as pd
from loguru import logger

from src.utils.outputs_analysis import SPARQLUtils, load_qald_json


def process_mk_dataset(csv_in: str, json_in: str, csv_out: str):
    logger.info("Starting formatting for Streamlit Dashboard...")

    try:
        df = pd.read_csv(csv_in)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    qald_map = load_qald_json(json_in, lang='mk')
    df['clean_q'] = df['original_question'].astype(str).str.strip()

    def row_processor(row):
        q_text = row['clean_q']
        json_gold = qald_map.get(q_text, {})
        gold_query = str(row.get('gold_query', json_gold.get('gold_query', '')))
        gold_result = str(json_gold.get('gold_result', ''))

        gold_ids = SPARQLUtils.extract_ids_from_text(gold_query)
        rag_ids = SPARQLUtils.parse_list_string(row.get('candidate_ids', ''))
        llm_used_ids = SPARQLUtils.extract_ids_from_text(str(row.get('generated_query', '')))

        rag_recall = len(rag_ids.intersection(gold_ids)) / len(gold_ids) if gold_ids else 1.0
        llm_precision = len(llm_used_ids.intersection(gold_ids)) / len(llm_used_ids) if llm_used_ids else 0.0

        gen_res_set = SPARQLUtils.normalize_result_string(row.get('result', ''))
        gold_res_set = SPARQLUtils.normalize_result_string(gold_result)
        _, _, final_f1 = SPARQLUtils.calculate_f1(gen_res_set, gold_res_set)

        ner_f1 = 1.0 if len(str(row.get('ner', ''))) > 5 else 0.0

        # Safe float parsing
        try:
            match_ratio_float = float(eval(str(row.get('match_ratio', '0/1'))))
            kw_ratio_float = float(eval(str(row.get('keyword_match_ratio', '0/1'))))
        except:
            match_ratio_float, kw_ratio_float = 0.0, 0.0

        return pd.Series([
            gold_query, gold_result, round(rag_recall, 3), round(llm_precision, 3),
            round(final_f1, 3), ner_f1, match_ratio_float, kw_ratio_float
        ])

    logger.info("Calculating Dashboard Metrics (RAG, LLM, F1)...")
    df[['gold_query_clean', 'gold_result_clean', 'RAG Recall', 'LLM Precision',
        'Final F1 Score', 'NER Judge F1', 'id_match_float', 'kw_match_float']] = df.apply(row_processor, axis=1)

    cols_to_keep = [
        'original_question', 'generated_query', 'gold_query_clean', 'candidate_ids',
        'gold_wikidata_ids', 'result', 'gold_result_clean', 'RAG Recall',
        'LLM Precision', 'Final F1 Score', 'NER Judge F1', 'time'
    ]

    # Filter columns safely
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    final_df = df[existing_cols].rename(columns={'gold_query_clean': 'gold_query', 'gold_result_clean': 'gold_result'})

    final_df.to_csv(csv_out, index=False)
    logger.success(f"Dashboard dataset saved to: {csv_out}")


if __name__ == "__main__":
    CSV_INPUT = "../../results/benchmark/sparql_outputs_mk_with_analysis.csv"
    JSON_INPUT = "../../qald_10_with_mk.json"
    CSV_OUTPUT = "dashboard_ready_mk.csv"

    if Path(CSV_INPUT).exists() and Path(JSON_INPUT).exists():
        process_mk_dataset(CSV_INPUT, JSON_INPUT, CSV_OUTPUT)
    else:
        logger.error("Files not found. Check paths.")
