import pandas as pd

# ================= CONFIGURATION =================
# Path to your CSV file
INPUT_CSV = "sparql_outputs_en.csv"
OUTPUT_CSV = INPUT_CSV  # Overwrite the file (or change name to save copy)


# =================================================

def normalize_result(val):
    """
    Cleans the 'result' column:
    - [] or NaN -> "<null>"
    - True/False -> "true"/"false"
    """
    # 1. Handle NaN / None
    if pd.isna(val):
        return None

    s_val = str(val).strip()

    # 2. Handle Empty List or Empty String
    if s_val == "<null>" or s_val == "":
        return None

    # 3. Handle Booleans (Python style or numbers)
    if s_val == "True" or s_val == "1":
        return "true"
    if s_val == "False" or s_val == "0":
        return "false"

    return s_val


def main():
    print(f"📂 Loading: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("❌ File not found.")
        return

    if 'result' not in df.columns:
        print("❌ Column 'result' not found in CSV.")
        return

    print("🔄 Cleaning 'result' column...")

    # Apply the cleaning function
    df['result'] = df['result'].apply(normalize_result)

    # Save back
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Cleaned and saved to: {OUTPUT_CSV}")

    # Show stats
    null_count = (df['result'] == "<null>").sum()
    print(f"   Rows with <null> results: {null_count}")


if __name__ == "__main__":
    main()
