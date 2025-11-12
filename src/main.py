import asyncio
import csv
import os
from typing import Any

from langchain_core.messages import HumanMessage
from tqdm import tqdm

from src.agent.graph import create_sparql_agent
from src.agent.prompts import sparql_agent_instruction
from src.databases.qdrant.qdrant import qdrant_db
from src.dataset.qald_10 import load_qald_json
from src.http_client.session import close_session

CSV_HEADER = [
    "original_question",
    "rephrased_question",
    "ner",
    "candidates",
    "examples",
    "generated_query",
    "result",
    "time"
]


async def process_question_and_write_attempts(
        item: dict[str, Any],
        agent,
        csv_writer,
):
    """Streams the agent's execution and writes each tool attempt to the CSV."""
    question = item["question"]
    initial_state = {
        "messages": [HumanMessage(content=sparql_agent_instruction.format(user_task=question))],
        "original_question": question,
        "attempts": 0,
        "log_data": []
    }

    try:
        # Use .astream() to get the state after each step
        async for step in agent.astream(initial_state):
            # We are interested in the state *after* the tool_executor has run
            if "tool_executor" in step:
                state = step["tool_executor"]

                # The log_data is now a list of logs from the last step
                new_logs = state.get("log_data", [])

                for log_entry in new_logs:
                    if log_entry:
                        # Create a row based on the header
                        row = [log_entry.get(h, "") for h in CSV_HEADER]
                        csv_writer.writerow(row)

    except Exception as e:
        print(f"\nCaught a streaming error for question '{question}': {e}")
        error_row = {"original_question": question, "results": f"STREAMING FAILED: {e}"}
        csv_writer.writerow([error_row.get(h, "") for h in CSV_HEADER])


async def main():
    csv_file_name = '../results/benchmark/sparql_outputs_improved_linking.csv'
    file_exists = os.path.exists(csv_file_name)

    with open(csv_file_name, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(CSV_HEADER)

        benchmark_data = load_qald_json()

        print("Compiling the SPARQL agent...")
        sparql_agent = create_sparql_agent()
        print("Agent compiled successfully. Starting benchmark...")

        for item in tqdm(benchmark_data):
            await process_question_and_write_attempts(item, sparql_agent, writer)

    print(f"\n\nSCRIPT FINISHED. Results are in '{csv_file_name}'.")
    qdrant_db.client.close()
    await close_session()


if __name__ == '__main__':
    asyncio.run(main())
