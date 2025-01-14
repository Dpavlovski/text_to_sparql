import logging
import uuid

from tqdm import tqdm

from src.databases.qdrant.qdrant import QdrantDatabase
from src.dataset.lcquad2_0 import get_dataset


def embedd_dataset():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the embedding process...")

    qdb = QdrantDatabase()
    dataset = get_dataset()
    logging.info(f"Dataset loaded with {len(dataset)} records.")

    for row in tqdm(dataset, desc="Embedding and upserting records"):
        question = row.get("question")
        sparql_query = row.get("sparql_wikidata")

        if not question or not sparql_query:
            logging.error("Skipping record with missing question or SPARQL query.")
            continue

        try:
            qdb.embedd_and_upsert_record(
                value=question,
                collection_name="lcquad2_0",
                unique_id=str(uuid.uuid4()),
                metadata={"answer": sparql_query}
            )
        except Exception as e:
            logging.error(f"Error processing record with question: {question}. Error: {e}")

    logging.info("Embedding process completed successfully.")


embedd_dataset()
