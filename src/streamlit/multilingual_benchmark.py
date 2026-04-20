import glob
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

# ================= CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate: src/dashboard/ -> src/ -> root/ -> results/benchmark/with_neighbors
RESULTS_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "results", "benchmark", "with_neighbors", "processed"))

# --- MANUAL GERBIL SCORES ---
GERBIL_DATA = [
    {"model": "gpt-4.1-mini", "language": "en", "Gerbil F1": 0.5002},
    {"model": "gpt-4.1-mini", "language": "de", "Gerbil F1": 0.2340},
    {"model": "gpt-4.1-mini", "language": "zh", "Gerbil F1": 0.2288},
    {"model": "gpt-4.1-mini", "language": "ru", "Gerbil F1": 0.2257},
    {"model": "nemotron-3-nano-30b-a3", "language": "en", "Gerbil F1": 0.3416},
    {"model": "nemotron-3-nano-30b-a3", "language": "de", "Gerbil F1": 0.2836},
    {"model": "nemotron-3-nano-30b-a3", "language": "zh", "Gerbil F1": 0.1082},
    {"model": "nemotron-3-nano-30b-a3", "language": "ru", "Gerbil F1": 0.2834}
]
# =================================================

st.set_page_config(page_title="Text-to-SPARQL Benchmark", layout="wide")


def parse_filename(filename):
    name = Path(filename).stem
    parts = name.split('_')

    lang = "unknown"
    model = "unknown"

    for p in parts:
        if len(p) == 2 and p in ['en', 'de', 'ru', 'zh']:
            lang = p
            break

    for p in parts:
        if any(k in p.lower() for k in ["gpt", "llama", "qwen", "mistral", "deepseek", "claude", "nemotron", "gemini"]):
            model = p
            break

    if model == "unknown" and len(parts) > 2:
        model = parts[-1]

    return lang, model


@st.cache_data
def load_data():
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))

    if not all_files:
        st.error(f"No CSV files found in: {RESULTS_DIR}")
        return pd.DataFrame()

    combined_df = []

    for f in all_files:
        try:
            df = pd.read_csv(f)
            # Make sure we load the standard metrics or the NER evaluation files
            if 'keyword_match_ratio' not in df.columns and 'ner_f1' not in df.columns:
                continue
            lang, model = parse_filename(f)
            df['language'] = lang
            df['model'] = model
            combined_df.append(df)
        except Exception as e:
            st.warning(f"Could not read {os.path.basename(f)}: {e}")

    if not combined_df:
        return pd.DataFrame()

    full_df = pd.concat(combined_df, ignore_index=True)

    # ADDED NER METRICS HERE (Removed res_f1 from requirement)
    numeric_cols = ['id_match_score', 'keyword_match_ratio', 'time', 'ner_precision', 'ner_recall', 'ner_f1']
    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    return full_df


def main():
    st.title("📊 Multilingual Text-to-SPARQL Benchmark")

    df = load_data()
    if df.empty:
        st.warning("No valid benchmark data found.")
        return

    # --- FILTERS ---
    st.sidebar.header("Filters")
    avail_langs = sorted(df['language'].unique())
    avail_models = sorted(df['model'].unique())

    sel_langs = st.sidebar.multiselect("Languages", avail_langs, default=avail_langs)
    sel_models = st.sidebar.multiselect("Models", avail_models, default=avail_models)

    filtered_df = df[(df['language'].isin(sel_langs)) & (df['model'].isin(sel_models))]

    if filtered_df.empty:
        st.info("No data matches the filters.")
        return

    # --- AGGREGATION ---
    agg_df = filtered_df.groupby(['model', 'language']).agg({
        'id_match_score': 'mean',
        'keyword_match_ratio': 'mean',
        'time': 'mean',
        'result': 'count'
    }).reset_index()

    agg_df = agg_df.rename(columns={
        'id_match_score': 'Entity Linking',
        'keyword_match_ratio': 'Keyword Match',
        'time': 'Avg Time (s)',
        'result': 'Samples'
    })

    # Merge GERBIL Scores
    gerbil_df = pd.DataFrame(GERBIL_DATA)
    agg_df = pd.merge(agg_df, gerbil_df, on=['model', 'language'], how='left')
    agg_df['Gerbil F1'] = agg_df['Gerbil F1'].fillna(0)

    # --- TOP ROW METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    if not agg_df.empty:
        best_idx = agg_df['Gerbil F1'].idxmax()
        best_row = agg_df.loc[best_idx]
        col1.metric("🏆 Top Performer (Gerbil F1)", f"{best_row['model']} ({best_row['language']})",
                    f"F1: {best_row['Gerbil F1']:.3f}")

    col2.metric("Total Samples", len(filtered_df))
    col3.metric("Avg Latency", f"{filtered_df['time'].mean():.2f}s")

    success_mask = (filtered_df['result'].notna()) & (filtered_df['result'] != "[]") & \
                   (filtered_df['result'] != "") & (filtered_df['result'] != "MISSING_IN_LOG")
    success_rate = success_mask.sum() / len(filtered_df)
    col4.metric("Success Rate (Returned Data)", f"{success_rate:.1%}")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Comparison", "🔍 Detail View", "⚡ Performance", "🧠 NER (English)"])

    with tab1:
        st.markdown("### 🏆 Result Accuracy")

        # --- ROW 1: Gerbil F1 Score ---
        st.caption("Gerbil F1 (Official Benchmark)")
        fig_gerbil = px.bar(agg_df, x="language", y="Gerbil F1", color="model", barmode="group",
                            text_auto='.2f', range_y=[0, 1.1])
        st.plotly_chart(fig_gerbil, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧩 Component Analysis")

        # --- ROW 2: Components ---
        r2_c1, r2_c2 = st.columns(2)

        with r2_c1:
            st.caption("Entity Linking Score (Did we find the right QID?)")
            fig_el = px.bar(agg_df, x="language", y="Entity Linking", color="model", barmode="group",
                            text_auto='.2f', range_y=[0, 1.1])
            st.plotly_chart(fig_el, use_container_width=True)

        with r2_c2:
            st.caption("Keyword Match Ratio (NER Performance)")
            fig_kw = px.bar(agg_df, x="language", y="Keyword Match", color="model", barmode="group",
                            text_auto='.2f', range_y=[0, 1.1])
            st.plotly_chart(fig_kw, use_container_width=True)

        # Radar Chart
        st.markdown("---")
        st.subheader("Metric Composition (Radar Chart)")
        radar_cols = ['Gerbil F1', 'Entity Linking', 'Keyword Match']
        fig_radar = go.Figure()
        for idx, row in agg_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[c] for c in radar_cols],
                theta=radar_cols,
                fill='toself',
                name=f"{row['model']} ({row['language']})"
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.subheader("Leaderboard")
        style_df = agg_df.sort_values("Gerbil F1", ascending=False)
        st.dataframe(
            style_df.style.background_gradient(subset=['Gerbil F1', 'Entity Linking', 'Keyword Match'], cmap="Greens"),
            width="stretch"
        )
        with st.expander("See Raw Data"):
            st.dataframe(filtered_df, width="stretch")

    with tab3:
        st.subheader("Time vs. Accuracy Tradeoff")
        fig_scat = px.scatter(
            agg_df, x="Avg Time (s)", y="Gerbil F1",
            color="model", symbol="language", size="Samples",
            hover_data=["model", "language"],
            title="Ideal: Top Left (High F1, Low Time)"
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    # --- NEW NER TAB ---
    with tab4:
        st.subheader("🧠 Semantic NER Evaluation (English Only)")
        st.markdown(
            "This tab visualizes the **LLM-as-a-Judge** evaluation, determining if the agent's extracted keywords "
            "semantically match the canonical labels required by the Gold SPARQL."
        )

        if 'ner_f1' not in filtered_df.columns:
            st.warning(
                "No NER evaluation columns (`ner_f1`, `ner_precision`) found in the data. Did you run the LLM judge script?")
        else:
            # Filter specifically for English datasets that have NER data
            ner_df = filtered_df[(filtered_df['language'] == 'en') & (filtered_df['ner_f1'] > 0)]

            if ner_df.empty:
                st.info(
                    "No English LLM-as-a-Judge NER data loaded. Ensure your `en_ner_judged_f1.csv` is in the processed directory.")
            else:
                # Group metrics by model
                ner_agg = ner_df.groupby('model').agg({
                    'ner_precision': 'mean',
                    'ner_recall': 'mean',
                    'ner_f1': 'mean'
                }).reset_index()

                # Melt data for grouped bar chart
                ner_melted = ner_agg.melt(
                    id_vars='model',
                    value_vars=['ner_precision', 'ner_recall', 'ner_f1'],
                    var_name='Metric',
                    value_name='Score'
                )

                # Rename metrics for clean display
                ner_melted['Metric'] = ner_melted['Metric'].replace({
                    'ner_precision': 'Precision (Cleanliness)',
                    'ner_recall': 'Recall (Completeness)',
                    'ner_f1': 'Macro F1'
                })

                # Plot
                fig_ner = px.bar(
                    ner_melted,
                    x='model',
                    y='Score',
                    color='Metric',
                    barmode='group',
                    text_auto='.2f',
                    range_y=[0, 1.1],
                    title="Semantic Extraction Accuracy (EN Subset)",
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
                )
                st.plotly_chart(fig_ner, use_container_width=True)

                # Data Table
                st.markdown("**Metric Summary**")
                display_ner = ner_agg.rename(columns={
                    'ner_precision': 'Precision (Cleanliness)',
                    'ner_recall': 'Recall (Completeness)',
                    'ner_f1': 'Macro F1'
                })
                st.dataframe(
                    display_ner.style.background_gradient(
                        subset=['Macro F1', 'Precision (Cleanliness)', 'Recall (Completeness)'], cmap="Blues"),
                    width="stretch"
                )


if __name__ == "__main__":
    main()