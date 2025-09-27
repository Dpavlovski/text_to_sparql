import logging
import uuid
from typing import List

from tqdm import tqdm

from src.databases.qdrant.qdrant import QdrantDatabase
from src.dataset.qald_10_results_embedings import extract_qald_query_ids
from src.utils.format_uri import extract_id_from_uri
from src.wikidata.api import get_wikidata_labels


def embedd_labels():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the embedding process...")

    qdb = QdrantDatabase()
    ids = extract_qald_query_ids()
    entity_ids: List[str] = [extract_id_from_uri(uri) for uri in ids]

    logging.info(f"Results loaded with {len(ids)} records.")

    labels_map = get_wikidata_labels(entity_ids)
    logging.info("Labels fetched.")

    for entity_id, labels in tqdm(labels_map.items(), desc="Embedding and upserting records"):
        for label in labels:
            if label["description"]:
                embedding_value = f"{label['label']} - {label['description']}"
            else:
                embedding_value = label["label"]

            try:
                qdb.embedd_and_upsert_record(
                    value=embedding_value,
                    collection_name="qald_10_labels",
                    unique_id=str(uuid.uuid4()),
                    metadata={"id": entity_id, "lang": label['language']}
                )
            except Exception as e:
                logging.error(f"Error processing label '{label}' for {entity_id}. Error: {e}")

    logging.info("Embedding process completed successfully.")


embedd_labels()
