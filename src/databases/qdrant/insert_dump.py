import json
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from qdrant_client import models
from tqdm import tqdm

from src.databases.qdrant.qdrant import QdrantDatabase
from src.llm.embed_labels import EmbeddingModel

BATCH_SIZE = 128
COLLECTION_NAME = "wikidata_labels_en"
VECTOR_SIZE = 384
NUM_WORKERS = 16


class Processor:
    def __init__(self):
        self.db = QdrantDatabase()
        self.embedder = EmbeddingModel()
        self._init_collection()

    def _init_collection(self):
        if not self.db.collection_exists(COLLECTION_NAME):
            self.db.create_collection(
                COLLECTION_NAME
            )


def process_file(file_pair: Tuple[Path, Path], lang: str = "en"):
    processor = Processor()
    label_file, desc_file = file_pair

    try:
        with open(label_file, 'r', encoding='utf-8') as lf, \
                open(desc_file, 'r', encoding='utf-8') as df:

            records = []
            for label_line, desc_line in zip(lf, df):
                label_data = json.loads(label_line)
                desc_data = json.loads(desc_line)

                value = f"{label_data.get(f'label_{lang}', '')} {desc_data.get(f'description_{lang}', '')}".strip()
                if value:
                    records.append((value, label_data['qid']))

            for i in range(0, len(records), BATCH_SIZE):
                batch = records[i:i + BATCH_SIZE]
                texts = [item[0] for item in batch]
                qids = [item[1] for item in batch]

                embeddings = processor.embedder.embed_batch(texts)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={"text": text, "lang": lang, "qid": qid}
                    ) for text, qid, emb in zip(texts, qids, embeddings)
                ]

                processor.db.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=False
                )

        return True
    except Exception as e:
        print(f"Failed {file_pair}: {traceback.format_exc()}")
        return False


def process_all_files(file_pairs: List[Tuple[Path, Path]]):
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_file, pair): pair for pair in file_pairs}

        with tqdm(total=len(file_pairs), desc="Processing") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    future.result()
                except Exception as e:
                    print(f"Critical error: {str(e)}")


if __name__ == "__main__":
    labels_dir = Path(
        "C:\\Users\\User\\PycharmProjects\\text_to_sparql\\src\\wikidata\\dump_processing\\data_processed\\labels")
    descriptions_dir = Path(
        "C:\\Users\\User\\PycharmProjects\\text_to_sparql\\src\\wikidata\\dump_processing\\data_processed\\descriptions")

    file_pairs = [
        (labels_dir / f"{i}.jsonl", descriptions_dir / f"{i}.jsonl")
        for i in range(2304)
        if (labels_dir / f"{i}.jsonl").exists()
           and (descriptions_dir / f"{i}.jsonl").exists()
    ]

    print(f"Found {len(file_pairs)} files to process")
    process_all_files(file_pairs)
