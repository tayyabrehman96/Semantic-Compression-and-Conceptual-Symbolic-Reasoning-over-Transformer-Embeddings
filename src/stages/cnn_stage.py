"""Stage: sequential Keras CNN configs on embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .. import config
from ..models_cnn import default_cnn_configs, train_cnn_config
from ..results_io import build_manifest, make_run_dir, save_run


def run_cnn_heads(
    emb_train,
    y_train,
    n_classes: int,
    class_names: list,
    embed_backend: str,
    kmeans: int,
    run_out_base: Path | None,
    extra_manifest: dict[str, Any] | None = None,
) -> tuple[Path, list[dict]]:
    if kmeans > 0:
        from ..clustering import augment_train_matrix_kmeans

        emb_train = augment_train_matrix_kmeans(emb_train, n_clusters=kmeans)

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            emb_train,
            y_train,
            test_size=0.1,
            random_state=config.RANDOM_STATE,
            stratify=y_train,
        )
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            emb_train,
            y_train,
            test_size=0.1,
            random_state=config.RANDOM_STATE,
            stratify=None,
        )

    run_dir = make_run_dir(run_out_base)
    manifest = build_manifest(
        {
            "mode": "cnn",
            "embed_backend": embed_backend,
            "kmeans_clusters": kmeans,
            "n_classes": n_classes,
            "class_names": class_names,
            **(extra_manifest or {}),
        }
    )
    cnn_rows = []
    for cfg in default_cnn_configs():
        row = train_cnn_config(
            cfg,
            X_tr,
            X_val,
            y_tr,
            y_val,
            num_classes=n_classes,
            epochs=config.CNN_EPOCHS,
            batch_size=config.CNN_BATCH_SIZE,
            seed=config.RANDOM_STATE,
        )
        cnn_rows.append(row)
    save_run(run_dir, manifest, cnn_rows, filename="metrics_cnn.csv")
    print(pd.DataFrame(cnn_rows).to_string(index=False))
    return run_dir, cnn_rows
