"""Stage: parallel sklearn heads on (optional K-Means augmented) embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..clustering import augment_embeddings_with_kmeans_distances
from ..models_sklearn import classifier_bank
from ..parallel_runner import run_sklearn_bank_parallel
from ..results_io import build_manifest, make_run_dir, save_run


def run_sklearn_heads(
    emb_train,
    emb_test,
    y_train,
    y_test,
    n_classes: int,
    class_names: list,
    kmeans: int,
    n_jobs: int | None,
    embed_backend: str,
    run_out_base: Path | None,
    extra_manifest: dict[str, Any] | None = None,
) -> tuple[Path, list[dict]]:
    if kmeans > 0:
        emb_train, emb_test = augment_embeddings_with_kmeans_distances(
            emb_train, emb_test, n_clusters=kmeans
        )
    run_dir = make_run_dir(run_out_base)
    manifest = build_manifest(
        {
            "mode": "sklearn",
            "embed_backend": embed_backend,
            "kmeans_clusters": kmeans,
            "n_classes": n_classes,
            "class_names": class_names,
            **(extra_manifest or {}),
        }
    )
    bank = classifier_bank(n_classes)
    rows = run_sklearn_bank_parallel(bank, emb_train, emb_test, y_train, y_test, n_jobs=n_jobs)
    save_run(run_dir, manifest, rows, filename="metrics_sklearn.csv")
    print(pd.DataFrame(rows).to_string(index=False))
    return run_dir, rows
