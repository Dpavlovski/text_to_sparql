import json
from pathlib import Path

import pandas as pd

# ================= CONFIGURATION =================
JSON_PATH = "../../qald_10_with_mk.json"
CSV_PATH = "../../results/benchmark/with_neighbors/processed/en_gpt-4.1-mini.csv"
OUTPUT_PATH = CSV_PATH
TARGET_LANG = "en"


# =================================================

def fix_missing_rows_keep_attempts():
    print(f"📂 Loading JSON: {JSON_PATH}")
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return

    # 1. Extract the "Gold Standard" questions
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

    # Ensure string matching works
    df_csv['original_question'] = df_csv['original_question'].astype(str).str.strip()

    print(f"   Found {len(df_csv)} rows in CSV.")

    # 3. Merge with INDICATOR
    print("🔄 Merging to find gaps...")

    df_final = pd.merge(
        df_expected,
        df_csv,
        on='original_question',
        how='left',
        indicator=True
    )

    # 4. Identify Truly Missing Rows
    truly_missing_mask = df_final['_merge'] == 'left_only'
    missing_count = truly_missing_mask.sum()

    # --- CRITICAL FIX: DELETE THE CATEGORICAL COLUMN NOW ---
    # We must remove '_merge' before doing any fillna operations
    del df_final['_merge']

    if missing_count > 0:
        print(f"⚠️  Found {missing_count} TRULY missing questions (adding placeholders)...")

        cols_to_fix = ['generated_query', 'result', 'candidates', 'ner']
        for col in cols_to_fix:
            if col not in df_final.columns:
                df_final[col] = ""
            # Safely set the missing rows to empty strings
            df_final.loc[truly_missing_mask, col] = ""
    else:
        print("✅ No missing questions found (Structure is complete).")

    # 5. Cleanup remaining NaNs safely
    if 'json_sort_order' in df_final.columns:
        del df_final['json_sort_order']

    df_final = df_final.fillna("")

    # 6. Save
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Saved fixed dataset to: {OUTPUT_PATH}")
    print(f"   Final Row Count: {len(df_final)}")


if __name__ == "__main__":
    fix_missing_rows_keep_attempts()