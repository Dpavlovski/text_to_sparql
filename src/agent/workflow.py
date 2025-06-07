from langgraph.graph import StateGraph, END

from src.tools.tools import *

# --- WIRING THE GRAPH ---

# 1. Initialize the graph
workflow = StateGraph(GraphState)

# 2. Add the nodes
workflow.add_node("agent_router", agent_router_node)
workflow.add_node("extract_labels", extract_labels_node)
workflow.add_node("find_entities", find_entities_node)
workflow.add_node("find_examples", find_examples_node)
workflow.add_node("generate_sparql", generate_sparql_node)
workflow.add_node("execute_sparql", execute_sparql_node)
workflow.add_node("format_answer", format_answer_node)

# 3. Set the entry point
workflow.set_entry_point("agent_router")


# 4. Define the conditional routing logic
def route_logic(state: GraphState):
    """Determines the next node to visit based on the 'next_action' state."""
    action = state.get("next_action")
    if state.get("error"):
        # For simplicity, we end on error. A more complex graph could loop or retry.
        return END
    if action == "Extract_Labels":
        return "extract_labels"
    if action == "Find_Similar_Entities":
        return "find_entities"
    if action == "Find_Similar_Examples":
        return "find_examples"
    if action == "Generate_SPARQL":
        return "generate_sparql"
    if action == "Execute_SPARQL":
        return "execute_sparql"
    if action == "finish":
        return "format_answer"
    return END


# 5. Add the edges
workflow.add_conditional_edges(
    "agent_router",
    route_logic,
    {
        "extract_labels": "extract_labels",
        "find_entities": "find_entities",
        "find_examples": "find_examples",
        "generate_sparql": "generate_sparql",
        "execute_sparql": "execute_sparql",
        "format_answer": "format_answer",
        END: END,
    },
)

# After each tool runs, we loop back to the router to decide the next step
workflow.add_edge("extract_labels", "agent_router")
workflow.add_edge("find_entities", "agent_router")
workflow.add_edge("find_examples", "agent_router")
workflow.add_edge("generate_sparql", "agent_router")
workflow.add_edge("execute_sparql", "agent_router")

# After formatting the answer, we end the process
workflow.add_edge("format_answer", END)

# 6. Compile the graph into a runnable app
app = workflow.compile()
