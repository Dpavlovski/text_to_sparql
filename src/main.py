import asyncio
import csv
import os
from typing import Any

from langchain_core.messages import HumanMessage
from loguru import logger
from tqdm import tqdm

from config.settings import settings
from src.agent.builder import SparqlAgentBuilder
from src.agent.prompts import sparql_agent_instruction
from src.config.config import BenchmarkConfig
from src.databases.qdrant.qdrant import QdrantDatabase
from src.dataset.qald_10 import load_qald_json
from src.http_client.session import close_session
from src.llm.llm_provider import LLMProvider

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
        csv_file,
        language: str,
):
    """Streams the agent's execution and writes each tool attempt to the CSV."""

    question = item.get("question", "Unknown Question")

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

                csv_file.flush()

    except Exception as e:
        logger.exception(f"Caught a streaming error for question '{question}'")
        error_row = {"original_question": question, "result": f"STREAMING FAILED: {e}"}
        row = [error_row.get(h, "") for h in CSV_HEADER]
        csv_writer.writerow(row)
        csv_file.flush()


async def main():
    qdrant_service = None

    try:
        config = BenchmarkConfig(TARGET_LANGUAGE)
        logger.info(f"Starting benchmark for language: {config.language.upper()}")
        logger.info(f"Using label collection: {config.get_collection_name('labels')}")

        # 1. Initialize Qdrant Service locally
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        qdrant_service = QdrantDatabase(host=qdrant_host, port=qdrant_port)

        # 2. Initialize LLM Service
        llm_service = LLMProvider.get_model(settings.default_llm_model)

        # 3. Inject Services into Builder
        logger.info("Compiling the SPARQL agent...")
        builder = SparqlAgentBuilder(llm_service=llm_service, qdrant_service=qdrant_service)
        sparql_agent = builder.compile()
        logger.success("Agent compiled successfully.")

        # 4. Prepare CSV Output
        csv_file_name = f'../results/benchmark/{config.language}_v2.csv'
        os.makedirs(os.path.dirname(csv_file_name), exist_ok=True)
        file_exists = os.path.exists(csv_file_name)

        benchmark_data = load_qald_json(lang=config.language)

        with open(csv_file_name, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(CSV_HEADER)

            logger.info("Starting processing...")
            for item in tqdm(benchmark_data[:], desc="Benchmarking"):
                await process_question_and_write_attempts(
                    item=item,
                    agent=sparql_agent,
                    csv_writer=writer,
                    csv_file=f,  # Pass the file handle so we can flush()
                    language=config.language
                )

        logger.success(f"SCRIPT FINISHED. Results are in '{csv_file_name}'.")

    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
    except Exception as e:
        logger.exception("A critical error occurred during the benchmark.")

    finally:
        # 5. Clean up resources reliably
        logger.info("Cleaning up resources...")
        if qdrant_service:
            await qdrant_service.close()
        await close_session()
        logger.info("Cleanup complete.")


if __name__ == '__main__':
    asyncio.run(main())
