import glob
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ================= CONFIGURATION =================
# Path to the folder containing your ANALYSIS csv files
# Files must be named: anything_{lang}_{model}.csv (e.g., analysis_en_gpt4.csv)
RESULTS_DIR = "../../results/benchmark"
# =================================================

st.set_page_config(page_title="Text-to-SPARQL Benchmark", layout="wide")


def parse_filename(filename):
    """
    Extracts Language and Model from filename.
    Expected format: *_{lang}_{model}.csv
    """
    name = Path(filename).stem
    parts = name.split('_')

    # Heuristic: Assuming format like 'sparql_outputs_en_v3_FINAL_ANALYSIS'
    # You might need to adjust this depending on your exact naming convention
    # Let's assume the user renames files to: results_en_gpt4.csv for simplicity
    # If not, we try to guess.

    try:
        # Try to find lang code (2 chars)
        lang = next((p for p in parts if len(p) == 2 and p in ['en', 'de', 'ru', 'zh', 'mk']), "unknown")

        # Assume model is the part after lang, or the last meaningful part
        # You can customize this logic
        model = "gpt-4.1-mini"  # Default for now if not found in name
        for p in parts:
            if "gpt" in p or "llama" in p or "claude" in p:
                model = p
                break

        return lang, model
    except:
        return "unknown", "unknown"


@st.cache_data
def load_data():
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*FINAL_ANALYSIS.csv"))

    if not all_files:
        st.error(f"No files found in {RESULTS_DIR}. Please run the analysis pipeline first.")
        return pd.DataFrame()

    combined_df = []

    for f in all_files:
        df = pd.read_csv(f)

        # parsing metadata from filename
        lang, model = parse_filename(f)

        # If your CSV has these columns already, prefer them. If not, use filename info.
        if 'language' not in df.columns:
            df['language'] = lang
        if 'model' not in df.columns:
            df['model'] = model

        combined_df.append(df)

    if not combined_df:
        return pd.DataFrame()

    full_df = pd.concat(combined_df, ignore_index=True)

    # Ensure numeric columns
    numeric_cols = ['res_f1', 'res_precision', 'res_recall', 'id_match_score', 'keyword_match_ratio', 'time']
    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    return full_df


def main():
    st.title("📊 Multilingual Text-to-SPARQL Benchmark")

    df = load_data()
    if df.empty:
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    languages = st.sidebar.multiselect("Select Languages", options=df['language'].unique(),
                                       default=df['language'].unique())
    models = st.sidebar.multiselect("Select Models", options=df['model'].unique(), default=df['model'].unique())

    # Filter Data
    filtered_df = df[(df['language'].isin(languages)) & (df['model'].isin(models))]

    if filtered_df.empty:
        st.warning("No data based on filters.")
        return

    # --- AGGREGATE STATS ---
    # Group by Model and Language
    agg_df = filtered_df.groupby(['model', 'language']).agg({
        'res_f1': 'mean',
        'res_precision': 'mean',
        'res_recall': 'mean',
        'id_match_score': 'mean',
        'keyword_match_ratio': 'mean',
        'time': 'mean',
        'result': 'count'  # Count total queries
    }).reset_index()

    # Rename for display
    agg_df = agg_df.rename(columns={'res_f1': 'F1 Score', 'time': 'Avg Time (s)', 'result': 'Samples'})

    # --- TOP METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    best_f1 = agg_df['F1 Score'].max()
    best_model_row = agg_df.loc[agg_df['F1 Score'].idxmax()]

    col1.metric("Best F1 Score", f"{best_f1:.2%}", f"{best_model_row['model']} ({best_model_row['language']})")
    col2.metric("Total Samples", len(filtered_df))
    col3.metric("Avg Execution Time", f"{filtered_df['time'].mean():.2f}s")

    # Success Rate (Queries that returned ANY result)
    # Assuming empty result is NaN or empty string
    success_count = filtered_df['result'].notna() & (filtered_df['result'] != "[]") & (filtered_df['result'] != "")
    success_rate = success_count.sum() / len(filtered_df)
    col4.metric("Execution Success Rate", f"{success_rate:.1%}")

    st.markdown("---")

    # --- TABS FOR VISUALIZATION ---
    tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "📈 Comparative Graphs", "🔍 Deep Dive"])

    with tab1:
        st.subheader("Model Performance Leaderboard")
        # Format table
        display_df = agg_df.copy()
        display_df['F1 Score'] = display_df['F1 Score'].apply(lambda x: f"{x:.3f}")
        display_df['id_match_score'] = display_df['id_match_score'].apply(lambda x: f"{x:.3f}")

        st.dataframe(
            display_df.sort_values(by="F1 Score", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.subheader("F1 Score by Language")
            fig_f1 = px.bar(
                agg_df,
                x="language",
                y="F1 Score",
                color="model",
                barmode="group",
                title="Result Accuracy (F1) across Languages",
                text_auto='.2f'
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        with col_g2:
            st.subheader("Entity Linking Accuracy")
            fig_el = px.bar(
                agg_df,
                x="language",
                y="id_match_score",
                color="model",
                barmode="group",
                title="Entity Linking Score (Did we find the right QIDs?)",
                text_auto='.2f'
            )
            st.plotly_chart(fig_el, use_container_width=True)

        # Radar Chart for detailed comparison
        st.subheader("Metric Balance Analysis")

        # Prepare data for Radar Chart (Normalize columns if needed)
        radar_cols = ['F1 Score', 'id_match_score', 'keyword_match_ratio', 'res_precision', 'res_recall']

        # Pivot for radar
        fig_radar = go.Figure()

        for index, row in agg_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[c] for c in radar_cols],
                theta=radar_cols,
                fill='toself',
                name=f"{row['model']} - {row['language']}"
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="Comparison of Metrics (Radar)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        st.subheader("Efficiency vs. Accuracy")
        fig_scatter = px.scatter(
            agg_df,
            x="Avg Time (s)",
            y="F1 Score",
            color="model",
            symbol="language",
            size="Samples",
            hover_data=['model', 'language'],
            title="Trade-off: Speed vs Accuracy (Top-Left is better)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Error Analysis (Empty Results)")

        # Calculate failure rate per group

        def calc_failure(x):
            fails = x.isna() | (x == "[]") | (x == "")
            return fails.sum() / len(x)

        fail_df = filtered_df.groupby(['model', 'language'])['result'].apply(calc_failure).reset_index(
            name="Failure Rate")

        fig_fail = px.bar(
            fail_df,
            x="language",
            y="Failure Rate",
            color="model",
            barmode="group",
            title="Percentage of Queries returning NO Results"
        )
        st.plotly_chart(fig_fail, use_container_width=True)


if __name__ == "__main__":
    main()
