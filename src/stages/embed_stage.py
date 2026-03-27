"""Stage 1: frozen encoders on train/test text splits."""

from __future__ import annotations

from ..embeddings import (
    encode_bert_mean_pool,
    encode_e5_multilingual,
    encode_sentence_transformer,
)


def embed_for_backend(backend: str, X_train: list[str], X_test: list[str]):
    if backend == "sentence_transformer":
        emb_tr = encode_sentence_transformer(X_train)
        emb_te = encode_sentence_transformer(X_test)
    elif backend == "bert_italian":
        emb_tr = encode_bert_mean_pool(X_train)
        emb_te = encode_bert_mean_pool(X_test)
    elif backend == "e5":
        emb_tr = encode_e5_multilingual(X_train)
        emb_te = encode_e5_multilingual(X_test)
    else:
        raise ValueError(f"Unknown embed backend: {backend}")
    return emb_tr, emb_te
