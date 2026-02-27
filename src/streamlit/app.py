import streamlit as st

# Page setup
st.set_page_config(page_title="SPARQL Agent Dashboard", layout="wide")

# FIX: Use filenames only, because app.py is in the same folder as these files
chat_page = st.Page("chat.py", title="Chat Agent", icon="💬")
outputs_page = st.Page("outputs.py", title="Output Analysis", icon="📊")
benchmark_page = st.Page("multilingual_benchmark.py", title="Benchmarks", icon="📈")

# Create navigation
pg = st.navigation([chat_page, outputs_page, benchmark_page])

# Run the selected page
pg.run()
