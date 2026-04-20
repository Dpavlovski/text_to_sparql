import os
import re
import sys

# Fix path to allow importing 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import asyncio
import streamlit as st
from langchain_core.messages import ToolMessage

# Import the correct builder and dependencies
from src.agent.builder import SparqlAgentBuilder
from src.llm.llm_provider import LLMProvider
from src.databases.qdrant.qdrant import QdrantDatabase
from src.config.settings import settings
from src.http_client.session import close_session

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
    """Helper to run async code safely in Streamlit"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


# --- Sidebar: Graph Visualization ---
with st.sidebar:
    st.header("⚙️ Architecture")

    # Generate and display the graph image dynamically
    try:
        # We only need the structure to draw the graph, so Qdrant can be None here
        llm_dummy = LLMProvider.get_model(settings.default_llm_model)
        builder_dummy = SparqlAgentBuilder(llm_service=llm_dummy, qdrant_service=None)
        app_graph = builder_dummy.compile()

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
    # 1. Initialize dependencies INSIDE the async loop to prevent Event Loop mismatch errors
    llm_service = LLMProvider.get_model(settings.default_llm_model)
    qdrant_service = QdrantDatabase(host=settings.qdrant_host, port=settings.qdrant_port)

    # 2. Compile the graph
    builder = SparqlAgentBuilder(llm_service=llm_service, qdrant_service=qdrant_service)
    app = builder.compile()

    initial_state = {
        "messages": [("user", user_input)],
        "attempts": 0,
        "original_question": user_input,
        "log_data": []
    }

    # Container for streaming output
    status_container = st.status("Agent is thinking...", expanded=True)
    final_response = ""

    try:
        # We use .astream to get updates node by node
        async for output in app.astream(initial_state):
            for node_name, state_update in output.items():

                # --- LLM Node Visualization ---
                if node_name == "llm":
                    latest_msg = state_update["messages"][-1]
                    if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                        status_container.write(f"🧠 **LLM decided:** Call tool `{latest_msg.tool_calls[0]['name']}`")
                    else:
                        status_container.write("🧠 **LLM decided:** Final answer ready.")
                        final_response = latest_msg.content

                # --- Tool Executor Visualization ---
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

                # --- Validator Visualization ---
                elif node_name == "validator":
                    messages = state_update.get("messages", [])
                    if messages and "Results are valid" in messages[0].content or "validated and appear correct" in \
                            messages[0].content:
                        status_container.info("⚖️ **Validation:** Results confirmed.")
                    else:
                        status_container.warning("⚖️ **Validation:** Results rejected. Looping back...")

        status_container.update(label="Processing Complete", state="complete", expanded=False)
        return final_response, initial_state

    finally:
        # Crucial Cleanup: Close async connections so they don't hang when Streamlit reruns
        await qdrant_service.close()
        await close_session()


# --- User Input ---
if prompt := st.chat_input("Ex: What are the child companies of Google?"):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Run Agent
    with st.chat_message("assistant"):
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
