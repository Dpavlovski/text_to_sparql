import ast
import json
import os
import re
from collections import defaultdict

import pandas as pd

import streamlit as st

st.set_page_config(layout="wide", page_title="SPARQL Agent Analysis")

st.markdown("""
<style>
    .reportview-container { margin-top: -2em; }
    .stDeployButton {display:none;}
    .stCodeBlock { border: 1px solid #ddd; }
    button[data-baseweb="tab"] { font-weight: bold; }
    .chip {
        display: inline-block; padding: 4px 10px; border-radius: 16px;
        font-size: 0.85em; margin-right: 6px; margin-bottom: 6px; font-weight: 500;
    }
    .chip-match { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; } 
    .chip-miss { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; } 
    .chip-extra { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; } 
    .result-box {
        padding: 10px; border-radius: 5px; background-color: #f1f3f5; border: 1px solid #ced4da;
        font-family: monospace; font-size: 0.9em; white-space: pre-wrap; max-height: 150px; overflow-y: auto; margin-bottom: 15px;
    }
    .gold-header { color: #856404; background-color: #fff3cd; padding: 5px 10px; border-radius: 5px; font-weight: bold; margin-bottom:5px; display:inline-block;}
    .gen-header { color: #0c5460; background-color: #d1ecf1; padding: 5px 10px; border-radius: 5px; font-weight: bold; margin-bottom:5px; display:inline-block;}
</style>
""", unsafe_allow_html=True)


def safe_parse_structure(x):
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
        return None, f"File not found at: {file_path}", None
    try:
        df = pd.read_csv(file_path)
        cols_to_float = ['res_f1', 'keyword_match_ratio', 'ner_f1', 'retrieval_recall_score', 'llm_precision_score']
        for col in cols_to_float:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        for col in ['messages', 'log_data']:
            if col in df.columns:
                df[col] = df[col].apply(safe_parse_structure)
        records = df.to_dict('records')
        grouped = defaultdict(list)
        for r in records:
            q_text = r.get('original_question', 'Unknown Question')
            grouped[q_text].append(r)
        return grouped, None, df
    except Exception as e:
        return None, str(e), None


def parse_list_col(data):
    if isinstance(data, list): return [str(x) for x in data]
    if isinstance(data, str):
        try:
            parsed = ast.literal_eval(data)
            if isinstance(parsed, list): return [str(x) for x in parsed]
        except:
            if "," in data and "[" not in data: return [x.strip() for x in data.split(",")]
    return []


def clean_sparql_prefixes(query):
    if not isinstance(query, str): return str(query)
    return re.sub(r'PREFIX\s+\w+:\s+<[^>]+>\s*', '', query, flags=re.IGNORECASE).strip()


def render_comparison_section(title, gold_list, gen_list, gold_title="🏆 Gold Standard", gen_title="🤖 Generated"):
    st.markdown(f"#### {title}")
    g_set = set([x.lower().strip() for x in parse_list_col(gold_list)])

    parsed_gen = parse_list_col(gen_list)
    clean_gen = []
    for item in parsed_gen:
        if "value=" in item:
            match = re.search(r"value='(.*?)'", item)
            clean_gen.append(match.group(1).lower().strip() if match else item.lower().strip())
        else:
            clean_gen.append(item.lower().strip())
    gen_set = set(clean_gen)

    matches = g_set.intersection(gen_set)
    missed = g_set.difference(gen_set)
    extras = gen_set.difference(g_set)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{gold_title}**")
        html = "".join([f"<span class='chip chip-match'>{i}</span>" for i in matches])
        html += "".join([f"<span class='chip chip-miss'>{i}</span>" for i in missed])
        st.markdown(html if html else "<span style='color:grey'>None</span>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"**{gen_title}**")
        html = "".join([f"<span class='chip chip-match'>{i}</span>" for i in matches])
        html += "".join([f"<span class='chip chip-extra'>{i}</span>" for i in extras])
        st.markdown(html if html else "<span style='color:grey'>None</span>", unsafe_allow_html=True)
    st.divider()


def render_metrics(record):
    f1 = float(record.get('res_f1', 0) or 0)
    ner_f1 = float(record.get('ner_f1', 0) or 0)
    rag_recall = float(record.get('retrieval_recall_score', 0) or 0)
    llm_prec = float(record.get('llm_precision_score', 0) or 0)
    kw_score = float(record.get('keyword_match_ratio', 0) or 0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Final F1", f"{f1:.2f}")
    c2.metric("NER Judge F1", f"{ner_f1:.2f}")
    c3.metric("RAG Recall", f"{rag_recall:.2f}")
    c4.metric("LLM Precision", f"{llm_prec:.2f}")
    c5.metric("Syntax", f"{kw_score:.2f}")

    status_bg = "#d4edda" if f1 == 1.0 else "#f8d7da"
    status_color = "#155724" if f1 == 1.0 else "#721c24"
    status_text = "Success" if f1 == 1.0 else "Failure"

    with c6:
        st.markdown(
            f"<div style='background-color:{status_bg}; color:{status_color}; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>{status_text}</div>",
            unsafe_allow_html=True)
    st.divider()


# --- Advanced Filter Logic ---
def filter_questions(grouped_data, filters):
    filtered = []

    for q_text, runs in grouped_data.items():
        last_run = runs[-1]
        f1 = float(last_run.get('res_f1', 0))
        ner_f1 = float(last_run.get('ner_f1', 0))
        rag_recall = float(last_run.get('retrieval_recall_score', 0))
        llm_prec = float(last_run.get('llm_precision_score', 0))
        result_text = str(last_run.get('result', ''))

        # Standard Range Filters
        if not (filters['f1'][0] <= f1 <= filters['f1'][1]): continue
        if not (filters['ner'][0] <= ner_f1 <= filters['ner'][1]): continue
        if not (filters['rag'][0] <= rag_recall <= filters['rag'][1]): continue
        if not (filters['llm'][0] <= llm_prec <= filters['llm'][1]): continue

        # Status Filter
        status = "Error/Empty"
        if result_text and result_text not in ["[]", "nan", ""]:
            status = "Correct" if f1 == 1.0 else "Wrong Answer"

        if status not in filters['status']: continue

        # Quick Diagnosis Presets
        preset = filters['preset']
        if preset == "RAG found it, LLM ignored it":
            if not (rag_recall > 0.8 and llm_prec < 0.5): continue
        elif preset == "NER failed completely":
            if not (ner_f1 < 0.3): continue
        elif preset == "Perfect NER, Failed RAG":
            if not (ner_f1 > 0.8 and rag_recall < 0.3): continue
        elif preset == "Syntax Perfect, Wrong Answer":
            kw_score = float(last_run.get('keyword_match_ratio', 0))
            if not (kw_score > 0.9 and f1 < 1.0): continue

        filtered.append(q_text)
    return filtered


# --- Main Logic ---
def main():
    st.title("🔎 Advanced Question Analysis")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(script_dir, "../../results/benchmark/with_neighbors/processed"))

    if not os.path.exists(results_dir):
        st.error(f"Results directory not found: {results_dir}")
        return

    csv_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.csv')])
    if not csv_files:
        st.warning("No CSV result files found.")
        return

    with st.sidebar:
        st.header("📂 Data Source")
        selected_filename = st.selectbox("Select file to analyze:", csv_files)
        st.divider()

    FILE_PATH = os.path.join(results_dir, selected_filename)
    grouped_data, error_msg, _ = load_data_grouped(FILE_PATH)

    if grouped_data is None:
        st.error(error_msg)
        return

    # --- ADVANCED SIDEBAR FILTERS ---
    with st.sidebar:
        st.header("🎯 Advanced Filters")

        # 1. Quick Presets (The "Aha!" buttons)
        preset = st.selectbox("⚡ Quick Diagnosis Presets", [
            "None",
            "RAG found it, LLM ignored it",
            "NER failed completely",
            "Perfect NER, Failed RAG",
            "Syntax Perfect, Wrong Answer"
        ], help="Instantly find specific failure cascades.")

        st.divider()

        # 2. Status
        status_opts = ["Correct", "Wrong Answer", "Error/Empty"]
        status_filter = st.multiselect("Result Status", status_opts, default=status_opts)

        # 3. Sliders
        with st.expander("🎚️ Fine-Tune Metrics", expanded=False):
            f1_range = st.slider("Final F1 Score", 0.0, 1.0, (0.0, 1.0), 0.1)
            ner_range = st.slider("NER Judge F1", 0.0, 1.0, (0.0, 1.0), 0.1)
            rag_range = st.slider("RAG Recall (Did DB find it?)", 0.0, 1.0, (0.0, 1.0), 0.1)
            llm_range = st.slider("LLM Precision (Did LLM use it?)", 0.0, 1.0, (0.0, 1.0), 0.1)

        filters = {
            'f1': f1_range, 'ner': ner_range, 'rag': rag_range,
            'llm': llm_range, 'status': status_filter, 'preset': preset
        }

        filtered_questions = filter_questions(grouped_data, filters)
        st.divider()
        st.write(f"Showing **{len(filtered_questions)}** / {len(grouped_data)} questions")

        if 'current_q_idx' not in st.session_state: st.session_state.current_q_idx = 0
        if st.session_state.current_q_idx >= len(filtered_questions):
            st.session_state.current_q_idx = 0

        if len(filtered_questions) > 0:
            selected_idx = st.number_input("Jump to Index", 1, len(filtered_questions),
                                           st.session_state.current_q_idx + 1)
            st.session_state.current_q_idx = selected_idx - 1
        else:
            st.warning("No questions match filters!")
            return

    def next_q():
        st.session_state.current_q_idx = min(len(filtered_questions) - 1, st.session_state.current_q_idx + 1)

    def prev_q():
        st.session_state.current_q_idx = max(0, st.session_state.current_q_idx - 1)

    c1, c2, c3 = st.columns([1, 6, 1])
    c1.button("⬅️ Prev", on_click=prev_q, use_container_width=True)
    c3.button("Next ➡️", on_click=next_q, use_container_width=True)

    current_q_text = filtered_questions[st.session_state.current_q_idx]
    runs = grouped_data[current_q_text]

    with c2:
        st.markdown(
            f"<h4 style='text-align: center;'>{st.session_state.current_q_idx + 1} / {len(filtered_questions)}</h4>",
            unsafe_allow_html=True)
        st.progress((st.session_state.current_q_idx + 1) / len(filtered_questions))

    st.divider()
    st.markdown(f"### ❓ {current_q_text}")

    tabs = st.tabs([f"Attempt {i + 1}" for i in range(len(runs))])

    for i, tab in enumerate(tabs):
        with tab:
            record = runs[i]
            render_metrics(record)

            with st.expander("⚔️ Queries & Results", expanded=True):
                st.markdown("<span class='gold-header'>🏆 Gold Standard</span>", unsafe_allow_html=True)
                st.code(clean_sparql_prefixes(record.get('gold_query', '# N/A')), language='sparql')
                st.markdown(f"**Expected Result:**")
                st.markdown(f"<div class='result-box'>{record.get('gold_result', 'None')}</div>",
                            unsafe_allow_html=True)
                st.divider()
                st.markdown("<span class='gen-header'>🤖 Generated</span>", unsafe_allow_html=True)
                st.code(clean_sparql_prefixes(record.get('generated_query', '# N/A')), language='sparql')
                st.markdown(f"**Actual Result:**")
                st.markdown(f"<div class='result-box'>{record.get('result', 'None')}</div>", unsafe_allow_html=True)

            if 'judge_gold_entities' in record and 'judge_agent_entities' in record and pd.notna(
                    record['judge_gold_entities']):
                render_comparison_section("🧠 NER Extraction Analysis (LLM Judge)",
                                          record.get('judge_gold_entities', []),
                                          record.get('judge_agent_entities', []),
                                          gold_title="🎯 Required Semantic Concepts",
                                          gen_title="🤖 Extracted by Agent")
                judge_reasoning = record.get('judge_reasoning', '')
                if pd.notna(judge_reasoning) and str(judge_reasoning).strip() != "":
                    st.info(f"**👨‍⚖️ Judge's Verdict:** {judge_reasoning}")

            render_comparison_section("📥 RAG Retrieval Analysis (All Fetched Candidates)",
                                      record.get('gold_wikidata_ids', []),
                                      record.get('retrieved_candidates', []),
                                      gold_title="🏆 Required IDs",
                                      gen_title="🔍 Found by Qdrant/API")

            render_comparison_section("🧠 LLM Execution Analysis (IDs actually used in SPARQL)",
                                      record.get('gold_wikidata_ids', []),
                                      record.get('used_candidate_ids', []),
                                      gold_title="🏆 Required IDs",
                                      gen_title="🤖 Synthesized by LLM")

            with st.expander("🌍 Raw Retrieved Context", expanded=False):
                st.markdown(record.get('candidates', 'No context.'))


if __name__ == "__main__":
    main()
