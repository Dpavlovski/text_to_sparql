from src.agent.graph import create_sparql_agent


def generate_graph_image():
    print("Compiling the agent...")
    # 1. Initialize the compiled graph
    app = create_sparql_agent()

    # 2. Get the graph object
    graph = app.get_graph()

    print("Generating Mermaid PNG...")
    try:
        # 3. Draw the graph as a PNG
        # draw_mermaid_png() renders the graph using Mermaid syntax
        image_data = graph.draw_mermaid_png()

        # 4. Save the binary data to a file
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
