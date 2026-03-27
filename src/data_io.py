"""Load DICE-style CSVs and produce stratified train/test splits with encoded labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config


def load_frame(path=None) -> pd.DataFrame:
    p = path or config.DATA_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at {p}.\n"
            f"  Expected canonical file: data/{config.STANDARD_DATASET_FILENAME}\n"
            "  Or set DICE_CSV to your CSV path. Required columns: "
            f"{config.TEXT_COLUMN!r}, {config.LABEL_COLUMN!r}."
        )
    df = pd.read_csv(p)
    for col in (config.TEXT_COLUMN, config.LABEL_COLUMN):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}. Found: {list(df.columns)}")
    return df


def train_test_texts_labels(df: pd.DataFrame | None = None):
    df = df if df is not None else load_frame()
    texts = df[config.TEXT_COLUMN].astype(str).tolist()
    labels = df[config.LABEL_COLUMN].tolist()
    strat = labels
    try:
        return train_test_split(
            texts,
            labels,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=strat,
        )
    except ValueError:
        return train_test_split(
            texts,
            labels,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=None,
        )


def prepare_supervised_split(df: pd.DataFrame | None = None):
    """Return texts, integer labels, and fitted LabelEncoder for sklearn / Keras sparse CE."""
    df = df if df is not None else load_frame()
    texts = df[config.TEXT_COLUMN].astype(str).tolist()
    raw_labels = df[config.LABEL_COLUMN].tolist()
    le = LabelEncoder()
    y = le.fit_transform(np.asarray(raw_labels))
    strat = y
    try:
        idx_train, idx_test = train_test_split(
            np.arange(len(texts)),
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=strat,
        )
    except ValueError:
        idx_train, idx_test = train_test_split(
            np.arange(len(texts)),
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=None,
        )
    X_train = [texts[i] for i in idx_train]
    X_test = [texts[i] for i in idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]
    return X_train, X_test, y_train, y_test, le
