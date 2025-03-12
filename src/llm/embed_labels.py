import os

from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel


def embed_labels(label: str) -> list[float] | None:
    load_dotenv()
    model_name = os.getenv("HF_MODEL")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cpu")
    model.eval()
    inputs = tokenizer(label, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].detach().squeeze(0).numpy()
    return vector.tolist()
