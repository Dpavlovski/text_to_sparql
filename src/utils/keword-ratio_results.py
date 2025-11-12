import pandas as pd

# Load your file
df = pd.read_csv("../../results/benchmark/sparql_outputs_mk_with_keyword_ratio.csv")

# Convert F1 column to numeric (in case it's text)
df["keyword_f1"] = pd.to_numeric(df["keyword_f1"], errors="coerce")

# Basic statistics
print("Average F1 Score:", df["keyword_f1"].mean())
print("Median F1 Score:", df["keyword_f1"].median())
print("Min F1 Score:", df["keyword_f1"].min())
print("Max F1 Score:", df["keyword_f1"].max())
print("Count:", len(df))

# F1 distribution
bins = [0, 0.25, 0.5, 0.75, 1.0]
labels = ["0–0.25 (Poor)", "0.25–0.5 (Fair)", "0.5–0.75 (Good)", "0.75–1.0 (Excellent)"]
df["F1 Category"] = pd.cut(df["keyword_f1"], bins=bins, labels=labels, include_lowest=True)
print("\nF1 Distribution:")
print(df["F1 Category"].value_counts().sort_index())
