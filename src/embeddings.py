"""Frozen transformer embeddings: SentenceTransformers (MiniLM / E5) and mean-pool BERT."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from . import config


def encode_sentence_transformer(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int | None = None,
    is_e5: bool = False,
) -> np.ndarray:
    name = model_name or config.ST_MODEL_NAME
    bs = batch_size if batch_size is not None else config.ST_ENCODE_BATCH_SIZE
    model = SentenceTransformer(name)
    to_encode = texts
    if is_e5 or "e5" in name.lower():
        to_encode = [f"passage: {t}" if not str(t).startswith("passage:") else t for t in texts]
    return model.encode(
        to_encode,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=bs,
    )


def encode_e5_multilingual(texts: list[str], model_name: str | None = None) -> np.ndarray:
    return encode_sentence_transformer(
        texts, model_name=model_name or config.E5_MODEL_NAME, is_e5=True
    )


def encode_bert_mean_pool(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
) -> np.ndarray:
    name = model_name or config.BERT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    chunks: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT batches"):
        batch = [str(t) for t in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = (summed / counts).cpu().numpy()
        chunks.append(pooled)
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    return np.vstack(chunks)
