"""Multi-process evaluation of sklearn heads via joblib (one job per classifier)."""

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone

from . import config
from .evaluation import evaluate_multiclass


def _fit_eval_one(
    name: str,
    estimator,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    clf = clone(estimator)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    metrics = evaluate_multiclass(y_test, preds)
    return {"classifier": name, **metrics}


def run_sklearn_bank_parallel(
    classifiers: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_jobs: int | None = None,
) -> list[dict[str, Any]]:
    """Train/eval each classifier in parallel processes (loky backend on Windows)."""
    jobs = n_jobs if n_jobs is not None else config.N_JOBS
    return Parallel(n_jobs=jobs)(
        delayed(_fit_eval_one)(name, est, X_train, X_test, y_train, y_test)
        for name, est in classifiers.items()
    )
