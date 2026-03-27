from __future__ import annotations

import argparse

import pandas as pd

from . import config
from .data_io import prepare_supervised_split
from .embeddings import encode_bert_mean_pool, encode_sentence_transformer
from .models_sklearn import classifier_bank
from .parallel_runner import run_sklearn_bank_parallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("sentence_transformer", "bert_italian"),
        default="sentence_transformer",
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, le = prepare_supervised_split()
    n_classes = len(le.classes_)

    if args.backend == "sentence_transformer":
        emb_train = encode_sentence_transformer(X_train)
        emb_test = encode_sentence_transformer(X_test)
    else:
        emb_train = encode_bert_mean_pool(X_train)
        emb_test = encode_bert_mean_pool(X_test)

    bank = classifier_bank(n_classes)
    rows = run_sklearn_bank_parallel(
        bank, emb_train, emb_test, y_train, y_test, n_jobs=args.n_jobs
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
