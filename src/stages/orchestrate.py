"""In-process full pipeline: embed → optional clustering inside stages → train heads."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data_io import prepare_supervised_split
from .cnn_stage import run_cnn_heads
from .embed_stage import embed_for_backend
from .sklearn_stage import run_sklearn_heads


def run_full_pipeline(
    mode: str,
    embed: str,
    kmeans: int,
    n_jobs: int | None,
    out_base: Path | None,
) -> Path:
    X_train, X_test, y_train, y_test, le = prepare_supervised_split()
    n_classes = len(le.classes_)
    class_names = le.classes_.tolist()
    emb_tr, emb_te = embed_for_backend(embed, X_train, X_test)

    if mode == "sklearn":
        run_dir, _ = run_sklearn_heads(
            emb_tr,
            emb_te,
            y_train,
            y_test,
            n_classes,
            class_names,
            kmeans,
            n_jobs,
            embed,
            out_base,
        )
    else:
        run_dir, _ = run_cnn_heads(
            emb_tr,
            y_train,
            n_classes,
            class_names,
            embed,
            kmeans,
            out_base,
        )
    print(f"\nSaved under: {run_dir}")
    return run_dir


def run_train_from_artifact(
    artifact: Path,
    mode: str,
    kmeans: int,
    n_jobs: int | None,
    out_base: Path | None,
) -> Path:
    from .artifact_io import load_embedding_artifact

    emb_tr, emb_te, y_train, y_test, meta = load_embedding_artifact(artifact)
    n_classes = int(meta["n_classes"])
    class_names = list(meta["class_names"])
    embed = meta.get("embed_backend", "sentence_transformer")

    if mode == "sklearn":
        run_dir, _ = run_sklearn_heads(
            emb_tr,
            emb_te,
            y_train,
            y_test,
            n_classes,
            class_names,
            kmeans,
            n_jobs,
            embed,
            out_base,
            extra_manifest={"from_artifact": str(artifact)},
        )
    else:
        run_dir, _ = run_cnn_heads(
            emb_tr,
            y_train,
            n_classes,
            class_names,
            embed,
            kmeans,
            out_base,
            extra_manifest={"from_artifact": str(artifact)},
        )
    print(f"\nSaved under: {run_dir}")
    return run_dir


def run_cache_embed_only(out: Path, embed: str) -> None:
    from .. import config
    from .artifact_io import save_embedding_artifact

    X_train, X_test, y_train, y_test, le = prepare_supervised_split()
    emb_tr, emb_te = embed_for_backend(embed, X_train, X_test)
    meta = {
        "embed_backend": embed,
        "class_names": le.classes_.tolist(),
        "n_classes": len(le.classes_),
        "data_csv": str(config.DATA_CSV),
        "seed": config.RANDOM_STATE,
    }
    save_embedding_artifact(out, emb_tr, emb_te, y_train, y_test, meta)
    print(f"Saved embedding artifact: {out}")
    print(pd.Series(meta).to_string())
