import ast
import json
import os
import re
from collections import defaultdict

import pandas as pd
import streamlit as st

# Page Config
st.set_page_config(layout="wide", page_title="SPARQL Agent Analysis")

# --- CSS Styling ---
st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    .stDeployButton {display:none;}
    .stCodeBlock { border: 1px solid #ddd; }

    /* Tab Header Styling */
    button[data-baseweb="tab"] {
        font-weight: bold;
    }

    /* Metrics Card Styling */
    .metric-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        background-color: #f9f9f9;
        text-align: center;
    }

    /* Comparison Chip Styling */
    .chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85em;
        margin-right: 6px;
        margin-bottom: 6px;
        font-weight: 500;
        border: 1px solid transparent;
    }

    .chip-match { background-color: #d4edda; color: #155724; border-color: #c3e6cb; } /* Green */
    .chip-miss { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb; } /* Red */
    .chip-extra { background-color: #fff3cd; color: #856404; border-color: #ffeeba; } /* Yellow */
    .chip-neutral { background-color: #e2e3e5; color: #383d41; border-color: #d6d8db; } /* Grey */

    .comparison-header {
        margin-top: 10px;
        margin-bottom: 5px;
        font-size: 1.1em;
        font-weight: bold;
        color: #444;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

def safe_parse_structure(x):
    """Parses string representations of lists/dicts from CSV."""
    if not isinstance(x, str): return x
    if pd.isna(x): return []
    try:
        return json.loads(x)
    except:
        pass
    try:
        return ast.literal_eval(x)
    except:
        pass
    return x


def load_data_grouped(file_path):
    if not os.path.exists(file_path):
        return None, f"File not found at: {file_path}"
    try:
        df = pd.read_csv(file_path)
        complex_cols = ['messages', 'log_data', 'history']
        for col in complex_cols:
            if col in df.columns:
                df[col] = df[col].apply(safe_parse_structure)
        records = df.to_dict('records')
        grouped = defaultdict(list)
        for r in records:
            q_text = r.get('original_question', 'Unknown Question')
            grouped[q_text].append(r)
        return grouped, None
    except Exception as e:
        return None, str(e)


def parse_list_col(data):
    """Helper to ensure we have a list of strings, handling various CSV formats."""
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, str):
        try:
            parsed = ast.literal_eval(data)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except:
            # Maybe it's a simple comma separated string?
            if "," in data and "[" not in data:
                return [x.strip() for x in data.split(",")]
    return []


def render_comparison_section(title, gold_list, gen_list):
    """Renders a side-by-side comparison with color coding."""
    st.markdown(f"#### {title}")

    # Normalize inputs
    g_set = set(parse_list_col(gold_list))
    # Parse generated list, handling potential Keywords objects if they exist
    parsed_gen = parse_list_col(gen_list)

    # Specialized cleanup for Keyword objects if they appear as dict strings
    clean_gen = []
    for item in parsed_gen:
        if "value=" in item:  # heuristic for Keyword(value='x') repr
            try:
                # distinct crude regex extraction if ast fails
                match = re.search(r"value='(.*?)'", item)
                if match:
                    clean_gen.append(match.group(1))
                else:
                    clean_gen.append(item)
            except:
                clean_gen.append(item)
        else:
            clean_gen.append(item)

    gen_set = set(clean_gen)

    # Set Logic
    matches = g_set.intersection(gen_set)
    missed = g_set.difference(gen_set)
    extras = gen_set.difference(g_set)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🏆 Gold Standard**")
        html = ""
        if not g_set:
            html = "<span style='color:#999; font-style:italic'>None provided</span>"
        else:
            for item in matches:
                html += f"<span class='chip chip-match' title='Matched'>{item}</span>"
            for item in missed:
                html += f"<span class='chip chip-miss' title='Missed by Agent'>{item}</span>"
        st.markdown(html, unsafe_allow_html=True)
        if missed:
            st.caption(f"Missed: {len(missed)}")

    with col2:
        st.markdown("**🤖 Agent Generated**")
        html = ""
        if not gen_set:
            html = "<span style='color:#999; font-style:italic'>None generated</span>"
        else:
            for item in matches:
                html += f"<span class='chip chip-match' title='Matched'>{item}</span>"
            for item in extras:
                html += f"<span class='chip chip-extra' title='Extra/Hallucinated'>{item}</span>"
        st.markdown(html, unsafe_allow_html=True)
        if extras:
            st.caption(f"Extras: {len(extras)}")

    # Visual Legend
    st.markdown("""
    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
        <span class="chip chip-match">Match</span>
        <span class="chip chip-miss">Missed (In Gold only)</span>
        <span class="chip chip-extra">Extra (In Generated only)</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()


def render_metrics(record):
    """Displays top-level metrics for the attempt."""
    # 1. Result F1
    f1 = record.get('res_f1', 0)
    try:
        f1 = float(f1)
    except:
        f1 = 0.0

    # 2. ID Match Score
    id_score = record.get('id_match_score', 0)
    try:
        id_score = float(id_score)
    except:
        id_score = 0.0

    # 3. Keyword Match Score (NEW)
    kw_score = record.get('keyword_match_ratio', 0)
    try:
        kw_score = float(kw_score)
    except:
        kw_score = 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("F1 Score", f"{f1:.2f}")
    with c2:
        st.metric("ID Match Score", f"{id_score:.2f}")
    with c3:
        st.metric("Keyword Match Score", f"{kw_score:.2f}")
    with c4:
        status = "Success" if f1 == 1.0 else "Failure"
        bg = "#d4edda" if f1 == 1.0 else "#f8d7da"
        color = "#155724" if f1 == 1.0 else "#721c24"
        st.markdown(
            f"<div style='background-color:{bg}; color:{color}; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>{status}</div>",
            unsafe_allow_html=True)

    st.divider()


# --- Main Logic ---

def main():
    st.title("🔎 SPARQL Agent Analysis")

    # ==============================
    # 🔧 PATH CONFIGURATION
    # ==============================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rel_path = "../../results/benchmark/sparql_outputs_en_v2-2_FINAL_ANALYSIS.csv"
    FILE_PATH = os.path.abspath(os.path.join(script_dir, rel_path))
    # ==============================

    grouped_data, error_msg = load_data_grouped(FILE_PATH)
    if grouped_data is None:
        st.error(f"❌ **Error Loading File**")
        st.warning(error_msg)
        st.stop()

    unique_questions = list(grouped_data.keys())

    # --- Sidebar ---
    with st.sidebar:
        st.header("Navigation")
        st.info(f"Loaded **{len(unique_questions)}** unique questions.")

        show_multi_only = st.checkbox("Show only multi-attempt questions")
        if show_multi_only:
            filtered_questions = [q for q in unique_questions if len(grouped_data[q]) > 1]
        else:
            filtered_questions = unique_questions

        st.divider()

        if 'current_q_idx' not in st.session_state:
            st.session_state.current_q_idx = 0

        total_q = len(filtered_questions)
        if total_q > 0:
            selected_idx = st.number_input("Jump to Index #", min_value=1, max_value=total_q,
                                           value=st.session_state.current_q_idx + 1)
            st.session_state.current_q_idx = selected_idx - 1

    if not filtered_questions:
        st.warning("No questions match the filter.")
        return

    def next_q():
        st.session_state.current_q_idx = min(len(filtered_questions) - 1, st.session_state.current_q_idx + 1)

    def prev_q():
        st.session_state.current_q_idx = max(0, st.session_state.current_q_idx - 1)

    col_prev, col_info, col_next = st.columns([1, 6, 1])
    with col_prev:
        st.button("⬅️ Prev", on_click=prev_q, use_container_width=True)
    with col_next:
        st.button("Next ➡️", on_click=next_q, use_container_width=True)

    current_q_text = filtered_questions[st.session_state.current_q_idx]
    runs = grouped_data[current_q_text]

    with col_info:
        current_display = st.session_state.current_q_idx + 1
        st.markdown(
            f"<h4 style='text-align: center; margin:0;'>Question {current_display} of {len(filtered_questions)}</h4>",
            unsafe_allow_html=True)
        st.progress(current_display / len(filtered_questions))

    st.divider()
    st.markdown(f"### ❓ {current_q_text}")

    tab_labels = [f"Attempt {i + 1}" for i in range(len(runs))]
    tabs = st.tabs(tab_labels)

    for i, tab in enumerate(tabs):
        with tab:
            record = runs[i]

            # 1. METRICS
            render_metrics(record)

            # 2. SPARQL COMPARISON
            with st.expander("⚔️ SPARQL Query Comparison", expanded=True):
                c_gold, c_gen = st.columns(2)
                with c_gold:
                    st.markdown("**🏆 Gold Query**")
                    st.code(record.get('gold_query', '# Not provided'), language='sparql')
                with c_gen:
                    st.markdown("**🤖 Generated Query**")
                    st.code(record.get('generated_query', '# Not generated'), language='sparql')

            st.divider()

            # 3. WIKIDATA ID COMPARISON
            render_comparison_section(
                "🆔 Wikidata ID Analysis",
                record.get('gold_wikidata_ids', []),
                record.get('candidate_ids', [])
            )

            # 4. KEYWORD COMPARISON
            render_comparison_section(
                "🔑 Keyword Analysis",
                record.get('gold_keywords', []),
                record.get('generated_keywords', [])
            )

            with st.expander("📊 View Raw JSON Data"):
                st.json(record)


if __name__ == "__main__":
    main()
