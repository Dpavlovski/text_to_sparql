import os
import sys

# Ensure Python can find the 'src' module if you run this script directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agent.builder import SparqlAgentBuilder
from src.llm.llm_provider import LLMProvider
from src.config.settings import settings


def generate_graph_image():
    print("Compiling the agent...")

    # 1. Initialize dependencies
    # We need a real LLM object because the builder calls `.bind_tools()` during __init__
    llm_service = LLMProvider.get_model(settings.default_llm_model)

    # Qdrant isn't actively queried during graph compilation, so passing None is perfectly safe here.
    qdrant_service = None

    # 2. Build and compile the graph
    builder = SparqlAgentBuilder(llm_service=llm_service, qdrant_service=qdrant_service)
    app = builder.compile()

    # 3. Get the graph object
    graph = app.get_graph()

    print("Generating Mermaid PNG...")
    try:
        # 4. Draw the graph as a PNG
        image_data = graph.draw_mermaid_png()

        # 5. Save the binary data to a file
        output_filename = "sparql_agent_workflow.png"
        with open(output_filename, "wb") as f:
            f.write(image_data)

        print(f"Success! Image saved to: {output_filename}")

    except Exception as e:
        print(f"Could not generate PNG directly: {e}")
        print("\n--- FALLBACK: MERMAID SYNTAX ---")
        print("Copy the code below and paste it into https://mermaid.live :")
        print("-" * 50)
        print(graph.draw_mermaid())
        print("-" * 50)


if __name__ == "__main__":
    generate_graph_image()
