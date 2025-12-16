import asyncio
import logging
import uuid

from datasets import load_from_disk
from tqdm import tqdm

from src.databases.qdrant.qdrant import qdrant_db
from src.llm.embed_labels import embed_value


async def embed_few_shot_examples():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the embedding process...")

    dataset = load_from_disk("../../dataset/lcquad2_ru")
    logging.info(f"Dataset loaded with {len(dataset)} records.")

    for row in tqdm(dataset, desc="Embedding and upserting records"):
        question = row.get("question_ru")
        sparql_query = row.get("sparql_wikidata")

        if not question or not sparql_query:
            logging.error("Skipping record with missing question or SPARQL query.")
            continue

        try:
            vector = embed_value(question)
            await qdrant_db.upsert_record(
                vector=vector,
                collection_name="lcquad2_0_ru",
                unique_id=str(uuid.uuid4()),
                payload={"answer": sparql_query, "value": question}
            )
        except Exception as e:
            logging.error(f"Error processing record with question: {question}. Error: {e}")

    logging.info("Embedding process completed successfully.")


if __name__ == "__main__":
    asyncio.run(embed_few_shot_examples())
