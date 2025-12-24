import json
from pathlib import Path

import pandas as pd

# ================= CONFIGURATION =================
# Update these paths to match your project
JSON_PATH = "../../qald_10_with_mk.json"
CSV_PATH = "../../results/benchmark/with_neighbors/sparql_outputs_ru_nemotron-3-nano-30b-a3.csv"
OUTPUT_PATH = CSV_PATH  # Overwrite the file, or change name to save separately
TARGET_LANG = "ru"


# =================================================

def fix_missing_rows_keep_attempts():
    print(f"📂 Loading JSON: {JSON_PATH}")
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return

    # 1. Extract the "Gold Standard" order of questions
    qald_questions = data.get('questions', data) if isinstance(data, dict) else data
    expected_data = []

    print(f"   Found {len(qald_questions)} questions in JSON.")

    for i, entry in enumerate(qald_questions):
        q_text = None
        for q in entry.get('question', []):
            if q.get('language') == TARGET_LANG:
                q_text = q.get('string', '').strip()
                break

        if q_text:
            # We add a 'sort_order' to force the final result to match JSON order
            expected_data.append({
                "original_question": q_text,
                "json_sort_order": i
            })

    df_expected = pd.DataFrame(expected_data)

    # 2. Load the existing Benchmark CSV
    print(f"📂 Loading CSV: {CSV_PATH}")
    if not Path(CSV_PATH).exists():
        print("❌ CSV file not found.")
        return

    df_csv = pd.read_csv(CSV_PATH)

    # Normalize string to ensure matching
    df_csv['original_question'] = df_csv['original_question'].astype(str).str.strip()

    # --- CRITICAL CHANGE: DO NOT DROP DUPLICATES ---
    # We want to keep every attempt.
    print(f"   Found {len(df_csv)} rows (attempts) in CSV.")

    # 3. Merge
    # A Left Merge on the JSON list will keep the JSON order.
    # If the CSV has 3 rows for Question 1, this merge will produce 3 rows in the result.
    # If the CSV has 0 rows for Question 2, this merge will produce 1 row (with NaNs).
    print("🔄 Merging to find gaps...")

    df_final = pd.merge(
        df_expected,
        df_csv,
        on='original_question',
        how='left'
    )

    # 4. Fill Missing Rows
    # Identify rows that were created by the merge but had no data in CSV
    # We check 'generated_query' (or 'result') being NaN
    missing_mask = df_final['generated_query'].isna()
    missing_count = missing_mask.sum()

    if missing_count > 0:
        print(f"⚠️  Found {missing_count} missing questions (filling placeholders)...")

        df_final.loc[missing_mask, 'result'] = "MISSING_IN_LOG"
        df_final.loc[missing_mask, 'generated_query'] = "ERROR: Script crashed or skipped"
        # Optional: Fill other columns with empty strings to look cleaner
        df_final.loc[missing_mask, 'candidates'] = ""
        df_final.loc[missing_mask, 'ner'] = ""
    else:
        print("✅ No missing questions found.")

    # 5. Cleanup
    if 'json_sort_order' in df_final.columns:
        del df_final['json_sort_order']

    # 6. Save
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Saved fixed dataset to: {OUTPUT_PATH}")
    print(f"   Final Row Count: {len(df_final)}")


if __name__ == "__main__":
    fix_missing_rows_keep_attempts()
