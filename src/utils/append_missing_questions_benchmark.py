import json
from pathlib import Path

import pandas as pd

# ================= CONFIGURATION =================
# Update these paths to match your project
JSON_PATH = "../../qald_10_with_mk.json"
CSV_PATH = "../../results/benchmark/sparql_outputs_en_v3_FINAL_ANALYSIS.csv"
OUTPUT_PATH = "../../results/benchmark/sparql_outputs_en_V3_FINAL_ANALYSIS.csv"
TARGET_LANG = "en"


# =================================================

def fix_missing_rows():
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
        # Find the question string for the target language
        for q in entry.get('question', []):
            if q.get('language') == TARGET_LANG:
                q_text = q.get('string', '').strip()
                break

        if q_text:
            # We keep the index to ensure order
            expected_data.append({"original_question": q_text, "json_id": entry.get('id')})

    # Create a DataFrame that represents the "Perfect" list
    df_expected = pd.DataFrame(expected_data)

    # 2. Load the existing Benchmark CSV
    print(f"📂 Loading CSV: {CSV_PATH}")
    if not Path(CSV_PATH).exists():
        print("❌ CSV file not found.")
        return

    df_csv = pd.read_csv(CSV_PATH)

    # Normalize question text in CSV to ensure matching works (trim spaces)
    df_csv['original_question'] = df_csv['original_question'].astype(str).str.strip()

    # Check for duplicates in CSV (e.g. if you ran the script twice)
    # We keep the LAST attempt if there are duplicates
    df_csv = df_csv.drop_duplicates(subset=['original_question'], keep='last')

    print(f"   Found {len(df_csv)} unique rows in CSV.")

    # 3. The Magic: Left Join
    # This aligns the CSV data to the JSON structure.
    # Rows missing in CSV will be created with NaN values.
    print("🔄 Merging and aligning rows...")

    df_final = pd.merge(
        df_expected,
        df_csv,
        on='original_question',
        how='left'
    )

    # 4. Fill Missing Rows
    # Check which rows have no 'result' (meaning they came from JSON but weren't in CSV)
    missing_mask = df_final['result'].isna() & df_final['generated_query'].isna()
    missing_count = missing_mask.sum()

    if missing_count > 0:
        print(f"⚠️  Found {missing_count} missing questions! filling placeholders...")

        # Fill specific columns to indicate error
        df_final.loc[missing_mask, 'result'] = "MISSING_IN_LOG"
        df_final.loc[missing_mask, 'generated_query'] = "ERROR: Script crashed or skipped"
    else:
        print("✅ No missing questions found. Dataset is complete.")

    # 5. Cleanup
    # Remove the temporary json_id helper column if you don't want it in the final CSV
    if 'json_id' in df_final.columns:
        del df_final['json_id']

    # 6. Save
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Saved fixed dataset to: {OUTPUT_PATH}")
    print(f"   Total Rows: {len(df_final)}")


if __name__ == "__main__":
    fix_missing_rows()
