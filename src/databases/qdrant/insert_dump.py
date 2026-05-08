import asyncio
import json
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from loguru import logger
from qdrant_client import models
from tqdm import tqdm

from databases.qdrant.embed_labels import EmbeddingModel
from src.config.settings import settings
from src.databases.qdrant.qdrant import QdrantDatabase

BATCH_SIZE = 128
COLLECTION_NAME = "wikidata_labels_en"
NUM_WORKERS = 16


async def async_process_file(file_pair: Tuple[Path, Path], lang: str = "en") -> bool:
    """Async worker function that actually does the embedding and uploading."""
    label_file, desc_file = file_pair

    qdrant_db = QdrantDatabase(host=settings.qdrant_host, port=settings.qdrant_port)
    embedder = EmbeddingModel()

    try:
        if not await qdrant_db.collection_exists(COLLECTION_NAME):
            await qdrant_db.create_collection(COLLECTION_NAME)

        with open(label_file, 'r', encoding='utf-8') as lf, \
                open(desc_file, 'r', encoding='utf-8') as df:

            records = []
            for label_line, desc_line in zip(lf, df):
                label_data = json.loads(label_line)
                desc_data = json.loads(desc_line)

                value = f"{label_data.get(f'label_{lang}', '')} {desc_data.get(f'description_{lang}', '')}".strip()
                if value:
                    records.append((value, label_data['qid']))

            # Process in batches
            for i in range(0, len(records), BATCH_SIZE):
                batch = records[i:i + BATCH_SIZE]
                texts = [item[0] for item in batch]
                qids = [item[1] for item in batch]

                embeddings = embedder.embed_batch(texts)

                points = [
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_OID, qid)),
                        vector=emb,
                        payload={"text": text, "lang": lang, "qid": qid}
                    ) for text, qid, emb in zip(texts, qids, embeddings)
                ]

                await qdrant_db.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=False
                )

        return True
    except Exception as e:
        logger.error(f"Failed {file_pair}: {traceback.format_exc()}")
        return False
    finally:
        await qdrant_db.close()


def process_file_sync_wrapper(file_pair: Tuple[Path, Path]) -> bool:
    """Synchronous wrapper to allow ProcessPoolExecutor to run the async code."""
    return asyncio.run(async_process_file(file_pair))


def process_all_files(file_pairs: List[Tuple[Path, Path]]):
    logger.info(f"Starting ProcessPoolExecutor with {NUM_WORKERS} workers.")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_file_sync_wrapper, pair): pair for pair in file_pairs}

        with tqdm(total=len(file_pairs), desc="Processing JSONL Dumps") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    success = future.result()
                    if not success:
                        logger.warning("A file pair failed to process.")
                except Exception as e:
                    logger.exception("Critical worker error")

# if __name__ == "__main__":
#     base_dir = Path("C:\\Users\\User\\PycharmProjects\\text_to_sparql\\src\\wikidata\\dump_processing\\data_processed")
#     labels_dir = base_dir / "labels"
#     descriptions_dir = base_dir / "descriptions"
#
#     file_pairs = [
#         (labels_dir / f"{i}.jsonl", descriptions_dir / f"{i}.jsonl")
#         for i in range(2304)
#         if (labels_dir / f"{i}.jsonl").exists() and (descriptions_dir / f"{i}.jsonl").exists()
#     ]
#
#     logger.info(f"Found {len(file_pairs)} valid file pairs to process")
#     if file_pairs:
#         process_all_files(file_pairs)
