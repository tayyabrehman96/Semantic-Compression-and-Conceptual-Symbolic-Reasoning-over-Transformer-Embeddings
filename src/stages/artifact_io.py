"""NPZ + JSON sidecar for cross-process stages (heavy models unloaded between steps)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_embedding_artifact(
    path: Path,
    emb_train: np.ndarray,
    emb_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    meta: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        emb_train=emb_train,
        emb_test=emb_test,
        y_train=y_train,
        y_test=y_test,
    )
    meta_path = path.parent / f"{path.stem}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_embedding_artifact(path: Path):
    path = Path(path)
    meta_path = path.parent / f"{path.stem}.meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    data = np.load(path)
    return (
        data["emb_train"],
        data["emb_test"],
        data["y_train"],
        data["y_test"],
        meta,
    )
