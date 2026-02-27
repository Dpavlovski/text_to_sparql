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
            if 'res_f1' not in df.columns:
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
    numeric_cols = ['res_f1', 'id_match_score', 'keyword_match_ratio', 'time']
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
        'res_f1': 'mean',
        'id_match_score': 'mean',
        'keyword_match_ratio': 'mean',
        'time': 'mean',
        'result': 'count'
    }).reset_index()

    agg_df = agg_df.rename(columns={
        'res_f1': 'Internal F1',
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
        best_idx = agg_df['Internal F1'].idxmax()
        best_row = agg_df.loc[best_idx]
        col1.metric("🏆 Top Performer (Internal)", f"{best_row['model']} ({best_row['language']})",
                    f"F1: {best_row['Internal F1']:.3f}")

    col2.metric("Total Samples", len(filtered_df))
    col3.metric("Avg Latency", f"{filtered_df['time'].mean():.2f}s")

    success_mask = (filtered_df['result'].notna()) & (filtered_df['result'] != "[]") & \
                   (filtered_df['result'] != "") & (filtered_df['result'] != "MISSING_IN_LOG")
    success_rate = success_mask.sum() / len(filtered_df)
    col4.metric("Success Rate", f"{success_rate:.1%}")

    st.markdown("---")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Comparison", "🔍 Detail View", "⚡ Performance"])

    with tab1:
        st.markdown("### 🏆 Result Accuracy")

        # --- ROW 1: F1 Scores ---
        r1_c1, r1_c2 = st.columns(2)

        with r1_c1:
            st.caption("Internal F1 (Calculated by our script)")
            fig_f1 = px.bar(agg_df, x="language", y="Internal F1", color="model", barmode="group",
                            text_auto='.2f', range_y=[0, 1.1])
            st.plotly_chart(fig_f1, use_container_width=True)

        with r1_c2:
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
        radar_cols = ['Internal F1', 'Gerbil F1', 'Entity Linking', 'Keyword Match']
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
        style_df = agg_df.sort_values("Internal F1", ascending=False)
        st.dataframe(
            style_df.style.background_gradient(subset=['Internal F1', 'Gerbil F1', 'Entity Linking'], cmap="Greens"),
            width="stretch"
        )
        with st.expander("See Raw Data"):
            st.dataframe(filtered_df, width="stretch")

    with tab3:
        st.subheader("Time vs. Accuracy Tradeoff")
        fig_scat = px.scatter(
            agg_df, x="Avg Time (s)", y="Internal F1",
            color="model", symbol="language", size="Samples",
            hover_data=["model", "language"],
            title="Ideal: Top Left (High F1, Low Time)"
        )
        st.plotly_chart(fig_scat, use_container_width=True)


if __name__ == "__main__":
    main()