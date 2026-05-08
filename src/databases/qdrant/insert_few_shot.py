import uuid

from datasets import load_from_disk
from loguru import logger
from tqdm import tqdm

from databases.qdrant.embed_labels import embed_value
from src.config.settings import settings
from src.databases.qdrant.qdrant import QdrantDatabase


async def embed_few_shot_examples():
    logger.info("Starting the few-shot embedding process...")

    dataset = load_from_disk("../../../lcquad_collections/lcquad2_ru")
    logger.info(f"Dataset loaded with {len(dataset)} records.")

    # Instantiate Qdrant locally
    qdrant_db = QdrantDatabase(host=settings.qdrant_host, port=settings.qdrant_port)

    try:
        for row in tqdm(dataset, desc="Embedding and upserting records"):
            question = row.get("question_ru")
            sparql_query = row.get("sparql_wikidata")

            if not question or not sparql_query:
                logger.warning("Skipping record with missing question or SPARQL query.")
                continue

            try:
                vector = embed_value(question)
                await qdrant_db.upsert_record(
                    unique_id=str(uuid.uuid4()),  # Few-shots don't have QIDs, random UUID is fine
                    collection_name="lcquad2_0_ru",
                    payload={"answer": sparql_query, "value": question},
                    vector=vector
                )
            except Exception as e:
                logger.error(f"Error processing record with question: '{question}'. Error: {e}")

        logger.success("Few-shot embedding process completed successfully.")

    finally:
        await qdrant_db.close()

# if __name__ == "__main__":
#     asyncio.run(embed_few_shot_examples())
