import os
import re
import sys

from langchain_core.messages import ToolMessage

# Fix path to allow importing 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import asyncio
import streamlit as st
from src.agent.graph import create_sparql_agent

# ... rest of the streamlit code provided previously
# --- Page Config ---
st.set_page_config(
    page_title="Wikidata SPARQL Agent",
    page_icon="🕸️",
    layout="wide"
)

# --- Custom CSS for SPARQL Highlighting ---
st.markdown("""
<style>
    .stCode { font-family: 'Fira Code', monospace; }
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .info { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
</style>
""", unsafe_allow_html=True)


# --- Async Helper ---
def run_async(coroutine):
    """Helper to run async code in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)


# --- Sidebar: Graph Visualization ---
with st.sidebar:
    st.header("⚙️ Architecture")

    # Generate and display the graph image dynamically
    try:
        app_graph = create_sparql_agent()
        png_data = app_graph.get_graph().draw_mermaid_png()
        st.image(png_data, caption="LangGraph Workflow")
    except Exception as e:
        st.error(f"Could not draw graph: {e}")

    st.markdown("---")
    st.markdown("**Legend:**")
    st.markdown("- **LLM Node**: Decides next steps")
    st.markdown("- **Tool Node**: Executes SPARQL/Validation")
    st.markdown("- **Validator**: Checks results quality")

# --- Main Interface ---
st.title("🕸️ Text-to-SPARQL Agent Explorer")
st.markdown("Ask a question to query Wikidata via natural language.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "code" in message:
            st.code(message["code"], language="sparql")
        if "expander_data" in message:
            with st.expander("View Debug Details"):
                st.json(message["expander_data"])


# --- Logic to Process Query ---
async def process_question(user_input):
    # Initialize the compiled graph
    app = create_sparql_agent()

    initial_state = {
        "messages": [("user", user_input)],
        "attempts": 0,
        "original_question": user_input,
        "log_data": []
    }

    # Container for streaming output
    status_container = st.status("Agent is thinking...", expanded=True)

    final_response = ""

    # We use .astream to get updates node by node
    async for output in app.astream(initial_state):
        for node_name, state_update in output.items():

            # --- 1. LLM Node Visualization ---
            if node_name == "llm":
                latest_msg = state_update["messages"][-1]
                if latest_msg.tool_calls:
                    status_container.write(f"🧠 **LLM decided:** Call tool `{latest_msg.tool_calls[0]['name']}`")
                else:
                    status_container.write("🧠 **LLM decided:** Final answer ready.")
                    final_response = latest_msg.content

            # --- 2. Tool Executor Visualization ---
            elif node_name == "tool_executor":
                messages = state_update.get("messages", [])
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        # Attempt to parse SPARQL from the tool output
                        sparql_match = re.search(r"(SELECT|ASK|DESCRIBE|CONSTRUCT).*?(?=\s*$)", msg.content,
                                                 re.DOTALL | re.IGNORECASE)

                        if sparql_match:
                            sparql_query = sparql_match.group(0)
                            status_container.markdown(f"🛠️ **Generated SPARQL:**")
                            status_container.code(sparql_query, language="sparql")

                        if "returned no results" in msg.content:
                            status_container.error("❌ Query returned no results. Retrying...")
                        elif "Tool call failed" in msg.content:
                            status_container.error(f"⚠️ Error: {msg.content}")
                        else:
                            status_container.success("✅ Tool executed successfully.")

            # --- 3. Validator Visualization ---
            elif node_name == "validator":
                messages = state_update.get("messages", [])
                if messages and "Results are valid" in messages[0].content:
                    status_container.info("⚖️ **Validation:** Results confirmed.")
                else:
                    status_container.warning("⚖️ **Validation:** Results rejected. Looping back...")

    status_container.update(label="Processing Complete", state="complete", expanded=False)
    return final_response, initial_state


# --- User Input ---
if prompt := st.chat_input("Ex: What are the child companies of Google?"):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run Agent
    with st.chat_message("assistant"):
        # We need a placeholder to stream the thought process
        try:
            response_text, final_state = run_async(process_question(prompt))

            # Display Final Answer
            if response_text:
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                fallback = "The agent finished but returned no text content."
                st.warning(fallback)
                st.session_state.messages.append({"role": "assistant", "content": fallback})

        except Exception as e:
            st.error(f"An error occurred: {e}")
