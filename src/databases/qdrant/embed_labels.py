import json
import traceback
import uuid
from pathlib import Path
from typing import List, Tuple

import torch
from loguru import logger
from qdrant_client import models
from transformers import AutoTokenizer, AutoModel

from src.config.settings import settings
from src.databases.qdrant.qdrant import QdrantDatabase


class EmbeddingModel:
    """Singleton-style wrapper for the HuggingFace embedding model."""

    def __init__(self):
        model_name = "intfloat/multilingual-e5-small"
        logger.debug(f"Loading embedding model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.debug("Embedding model loaded successfully.")

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
    """Helper function to embed a single string."""
    return embedder.embed_batch([value])[0]


BATCH_SIZE = 128
RESUME_FILE_NUM = 0
RESUME_LINE_NUM = 0


class Processor:
    """Manages the connection to Qdrant and the embedding model for batch processing."""

    def __init__(self, collection_name: str, vector_size: int, db: QdrantDatabase):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.db = db
        self.embedder = embedder  # Reuse the globally loaded model

    async def init_collection(self):
        """Asynchronously ensures the collection exists before inserting."""
        if not await self.db.collection_exists(self.collection_name):
            await self.db.create_collection(
                collection_name=self.collection_name,
                vector_size=self.vector_size,
                distance=models.Distance.COSINE
            )


async def process_batch(processor: Processor, records: List[Tuple[str, str]], lang: str):
    """Embeds and uploads a single batch of text to Qdrant."""
    texts = [item[0] for item in records]
    qids = [item[1] for item in records]

    embeddings = processor.embedder.embed_batch(texts)

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, qid)),
            vector=emb,
            payload={"text": text, "lang": lang, "qid": qid}
        ) for text, qid, emb in zip(texts, qids, embeddings)
    ]

    await processor.db.client.upsert(
        collection_name=processor.collection_name,
        points=points,
        wait=True
    )


async def process_file(file_pair: Tuple[Path, Path], processor: Processor, lang: str = "en") -> bool:
    """Reads a pair of JSONL files and processes them in batches."""
    label_file, desc_file = file_pair
    file_num = int(label_file.stem)

    if file_num < RESUME_FILE_NUM:
        logger.info(f"Skipping file {file_num} (before resume point)")
        return True

    try:
        with open(label_file, 'r', encoding='utf-8') as lf, \
                open(desc_file, 'r', encoding='utf-8') as df:

            records = []
            line_count = 0
            logger.info(f"Processing file {file_num}...")

            # Fast-forward if resuming
            if file_num == RESUME_FILE_NUM:
                for _ in range(RESUME_LINE_NUM):
                    next(lf)
                    next(df)
                    line_count += 1
                logger.info(f"Resumed from line {RESUME_LINE_NUM}")

            for label_line, desc_line in zip(lf, df):
                line_count += 1
                if line_count % 5000 == 0:
                    logger.debug(f"File {file_num}: Processed {line_count} lines...")

                label_data = json.loads(label_line)
                desc_data = json.loads(desc_line)

                label = label_data.get(f'label_{lang}')
                desc = desc_data.get(f'description_{lang}')

                if not label and not desc:
                    continue

                value = f"{label or ''} {desc or ''}".strip()
                records.append((value, label_data['qid']))

                if len(records) >= BATCH_SIZE:
                    await process_batch(processor, records, lang)
                    records = []

            if records:
                await process_batch(processor, records, lang)

        logger.success(f"Completed file {file_num} ({line_count} total lines)")
        return True

    except Exception as e:
        logger.error(f"Failed processing {file_pair}:\n{traceback.format_exc()}")
        return False


async def process_all_files(file_pairs: List[Tuple[Path, Path]], collection_name: str, vector_size: int):
    """Main orchestrator for processing all dump files sequentially."""

    qdrant_db = QdrantDatabase(host=settings.qdrant_host, port=settings.qdrant_port)
    processor = Processor(collection_name, vector_size, db=qdrant_db)

    try:
        await processor.init_collection()

        for pair in sorted(file_pairs, key=lambda x: int(x[0].stem)):
            file_num = int(pair[0].stem)

            if file_num < RESUME_FILE_NUM:
                continue

            success = await process_file(pair, processor, lang="en")
            if not success:
                logger.error(f"Aborting processing due to failure in file {file_num}")
                break

    finally:
        await qdrant_db.close()

# if __name__ == "__main__":
#     COLLECTION_NAME = "wikidata_labels_en"
#     VECTOR_SIZE = 384
#
#     base_dir = Path("/wikidata/dump_processing/data_processed")
#     labels_dir = base_dir / "labels"
#     descriptions_dir = base_dir / "descriptions"
#
#     file_pairs = [
#         (labels_dir / f"{i}.jsonl", descriptions_dir / f"{i}.jsonl")
#         for i in range(2304)
#         if (labels_dir / f"{i}.jsonl").exists() and (descriptions_dir / f"{i}.jsonl").exists()
#     ]
#
#     logger.info(f"Found {len(file_pairs)} files to process")
#
#     if file_pairs:
#         asyncio.run(process_all_files(file_pairs, COLLECTION_NAME, VECTOR_SIZE))
