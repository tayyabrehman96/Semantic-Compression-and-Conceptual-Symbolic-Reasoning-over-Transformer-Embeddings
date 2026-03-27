"""Central paths, dataset naming, seeds, and worker counts for reproducibility."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical on-disk name for a DICE-derived CSV export (symlink or copy allowed).
STANDARD_DATASET_FILENAME = "dice_crime_news_modena.csv"

_DEFAULT_DATA = REPO_ROOT / "data" / STANDARD_DATASET_FILENAME
_LEGACY_DATA = REPO_ROOT / "data" / "italian_crime_news.csv"

if os.environ.get("DICE_CSV"):
    DATA_CSV = Path(os.environ["DICE_CSV"])
elif _DEFAULT_DATA.exists():
    DATA_CSV = _DEFAULT_DATA
else:
    DATA_CSV = _LEGACY_DATA if _LEGACY_DATA.exists() else _DEFAULT_DATA

TEXT_COLUMN = os.environ.get("DICE_TEXT_COL", "text")
LABEL_COLUMN = os.environ.get("DICE_LABEL_COL", "word2vec_tag")

RANDOM_STATE = int(os.environ.get("DICE_SEED", "42"))
TEST_SIZE = float(os.environ.get("DICE_TEST_SIZE", "0.2"))

# Embedding backends (Hugging Face ids)
ST_MODEL_NAME = os.environ.get(
    "ST_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
BERT_MODEL_NAME = os.environ.get("BERT_MODEL", "dbmdz/bert-base-italian-cased")
E5_MODEL_NAME = os.environ.get(
    "E5_MODEL", "intfloat/multilingual-e5-large-instruct"
)

# Parallel sklearn / joblib (-1 = all cores)
N_JOBS = int(os.environ.get("DICE_N_JOBS", "-1"))

# Outputs
RESULTS_DIR = Path(os.environ.get("DICE_RESULTS_DIR", REPO_ROOT / "results" / "runs"))

# Training
CNN_EPOCHS = int(os.environ.get("DICE_CNN_EPOCHS", "15"))
CNN_BATCH_SIZE = int(os.environ.get("DICE_CNN_BATCH", "32"))
ST_ENCODE_BATCH_SIZE = int(os.environ.get("ST_BATCH_SIZE", "64"))
