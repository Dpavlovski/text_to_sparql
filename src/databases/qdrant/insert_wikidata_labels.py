from typing import List

from loguru import logger
from tqdm import tqdm

from databases.qdrant.embed_labels import embed_value
from src.config.settings import settings
from src.databases.qdrant.qdrant import QdrantDatabase
from src.dataset.qald_10_results_embedings import extract_qald_query_ids
from src.utils.format_uri import extract_id_from_uri
from src.wikidata.api import get_wikidata_labels


async def embed_labels():
    logger.info("Starting the Wikidata labels embedding process...")

    ids = extract_qald_query_ids()
    if not ids:
        logger.error("No QALD query IDs extracted. Aborting.")
        return

    entity_ids: List[str] = [extract_id_from_uri(uri) for uri in ids if extract_id_from_uri(uri)]
    logger.info(f"Results loaded with {len(entity_ids)} unique valid entity IDs.")

    labels_map = get_wikidata_labels(entity_ids, language="en")
    logger.info("Labels fetched from Wikidata API.")

    qdrant_db = QdrantDatabase(host=settings.qdrant_host, port=settings.qdrant_port)

    try:
        for entity_id, label_text in tqdm(labels_map.items(), desc="Embedding and upserting labels"):

            if not label_text:
                continue

            try:
                vector = embed_value(label_text)

                await qdrant_db.upsert_record(
                    unique_id=entity_id,
                    collection_name="qald_10_labels",
                    payload={"id": entity_id, "lang": "en", "text": label_text},
                    vector=vector
                )
            except Exception as e:
                logger.error(f"Error processing label '{label_text}' for {entity_id}. Error: {e}")

        logger.success("Label embedding process completed successfully.")

    finally:
        await qdrant_db.close()

# if __name__ == "__main__":
#     asyncio.run(embed_labels())
