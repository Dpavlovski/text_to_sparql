import asyncio
import csv
import os
from typing import Any

from langchain_core.messages import HumanMessage
from tqdm import tqdm

from src.agent.graph import create_sparql_agent
from src.agent.prompts import sparql_agent_instruction
from src.config.config import BenchmarkConfig
from src.databases.qdrant.qdrant import qdrant_db
from src.dataset.qald_10 import load_qald_json
from src.http_client.session import close_session

TARGET_LANGUAGE = "en"

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
        language: str,
):
    """Streams the agent's execution and writes each tool attempt to the CSV."""
    question = "Which country was Bill Gates born in?"  # item["question"]

    initial_state = {
        "messages": [HumanMessage(content=sparql_agent_instruction.format(user_task=question))],
        "original_question": question,
        "attempts": 0,
        "log_data": [],
        "language": language
    }

    try:
        async for step in agent.astream(initial_state):
            if "tool_executor" in step:
                state = step["tool_executor"]

                new_logs = state.get("log_data", [])

                for log_entry in new_logs:
                    if log_entry:
                        row = [log_entry.get(h, "") for h in CSV_HEADER]
                        csv_writer.writerow(row)

    except Exception as e:
        print(f"\nCaught a streaming error for question '{question}': {e}")
        error_row = {"original_question": question, "result": f"STREAMING FAILED: {e}"}
        row = [error_row.get(h, "") for h in CSV_HEADER]
        csv_writer.writerow(row)


async def main():
    try:
        config = BenchmarkConfig(TARGET_LANGUAGE)
        print(f"Starting benchmark for language: {config.language.upper()}")
        print(f"Using label collection: {config.get_collection_name('labels')}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    csv_file_name = f'../results/benchmark/{config.language}.csv'
    file_exists = os.path.exists(csv_file_name)

    with open(csv_file_name, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(CSV_HEADER)

        benchmark_data = load_qald_json(lang=config.language)

        print("Compiling the SPARQL agent...")
        sparql_agent = create_sparql_agent()
        print("Agent compiled successfully. Starting processing...")

        for item in tqdm(benchmark_data[:], desc="Benchmarking"):
            await process_question_and_write_attempts(
                item,
                sparql_agent,
                writer,
                language=config.language
            )

    print(f"\n\nSCRIPT FINISHED. Results are in '{csv_file_name}'.")

    try:
        await qdrant_db.client.close()
        await close_session()
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == '__main__':
    asyncio.run(main())
