import json
import traceback
import uuid
from pathlib import Path
from typing import List, Tuple

import torch
from qdrant_client import models
from transformers import AutoTokenizer, AutoModel

from src.databases.qdrant.qdrant import QdrantDatabase


class EmbeddingModel:
    def __init__(self):
        model_name = "intfloat/multilingual-e5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.extend([embedding.tolist() for embedding in batch_embeddings])

        return all_embeddings


embedder = EmbeddingModel()


def embed_value(value: str) -> List[float]:
    return embedder.embed_batch([value])[0]


BATCH_SIZE = 128
COLLECTION_NAME = "wikidata_labels_en"
VECTOR_SIZE = 384

# Resume constants
RESUME_FILE_NUM = 825  # File to resume from
RESUME_LINE_NUM = 362  # Line to resume from


class Processor:
    def __init__(self):
        self.db = QdrantDatabase()
        self.embedder = EmbeddingModel()
        self._init_collection()

    def _init_collection(self):
        if not self.db.collection_exists(COLLECTION_NAME):
            self.db.create_collection(
                COLLECTION_NAME,
                vector_size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )


def process_file(file_pair: Tuple[Path, Path], lang: str = "en"):
    processor = Processor()
    label_file, desc_file = file_pair
    file_num = int(label_file.stem)

    # Skip files before our resume point
    if file_num < RESUME_FILE_NUM:
        print(f"Skipping file {file_num} (before resume point)")
        return True

    try:
        with open(label_file, 'r', encoding='utf-8') as lf, \
                open(desc_file, 'r', encoding='utf-8') as df:

            records = []
            line_count = 0
            print(f"Processing file {file_num}...")

            # Skip lines if resuming this specific file
            if file_num == RESUME_FILE_NUM:
                for _ in range(RESUME_LINE_NUM):
                    next(lf)
                    next(df)
                    line_count += 1
                print(f"Resuming from line {RESUME_LINE_NUM}")

            for label_line, desc_line in zip(lf, df):
                line_count += 1
                if line_count % 1000 == 0:
                    print(f"Processed {line_count} lines...")

                label_data = json.loads(label_line)
                desc_data = json.loads(desc_line)

                label = label_data.get(f'label_{lang}')
                desc = desc_data.get(f'description_{lang}')

                if not label and not desc:
                    continue

                value = f"{label or ''} {desc or ''}".strip()
                records.append((value, label_data['qid']))

                if len(records) >= BATCH_SIZE:
                    process_batch(processor, records, lang)
                    records = []

            # Process remaining records
            if records:
                process_batch(processor, records, lang)

        print(f"Completed file {file_num} ({line_count} lines)")
        return True
    except Exception as e:
        print(f"Failed {file_pair}: {traceback.format_exc()}")
        return False


def process_batch(processor, records, lang):
    texts = [item[0] for item in records]
    qids = [item[1] for item in records]

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
        wait=True
    )


def process_all_files(file_pairs: List[Tuple[Path, Path]]):
    # Process files in order
    for pair in sorted(file_pairs, key=lambda x: int(x[0].stem)):
        file_num = int(pair[0].stem)

        # Skip files before resume point
        if file_num < RESUME_FILE_NUM:
            continue

        success = process_file(pair, "en")
        if not success:
            print(f"Aborting processing due to failure in file {file_num}")
            break


if __name__ == "__main__":
    labels_dir = Path("data_processed/labels")
    descriptions_dir = Path("data_processed/descriptions")

    file_pairs = [
        (labels_dir / f"{i}.jsonl", descriptions_dir / f"{i}.jsonl")
        for i in range(2304)
        if (labels_dir / f"{i}.jsonl").exists()
           and (descriptions_dir / f"{i}.jsonl").exists()
    ]

    print(f"Found {len(file_pairs)} files to process")
    process_all_files(file_pairs)
