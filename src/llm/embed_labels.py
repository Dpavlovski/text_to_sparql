import os
from typing import List

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    def __init__(self):
        load_dotenv()
        model_name = os.getenv("HF_MODEL")
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
