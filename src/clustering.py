"""K-Means on embeddings: distance features for interpretability / rare-label structure."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from . import config


def augment_embeddings_with_kmeans_distances(
    emb_train: np.ndarray,
    emb_test: np.ndarray,
    n_clusters: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate original embeddings with distances to cluster centers (train-fit, test-transform)."""
    rs = random_state if random_state is not None else config.RANDOM_STATE
    km = KMeans(n_clusters=n_clusters, random_state=rs, n_init=10)
    km.fit(emb_train)
    d_tr = km.transform(emb_train)
    d_te = km.transform(emb_test)
    return np.hstack([emb_train, d_tr]), np.hstack([emb_test, d_te])


def augment_train_matrix_kmeans(emb_train: np.ndarray, n_clusters: int, random_state: int | None = None):
    """Fit K-Means on train rows only and append distance features (for CNN train/val split)."""
    rs = random_state if random_state is not None else config.RANDOM_STATE
    km = KMeans(n_clusters=n_clusters, random_state=rs, n_init=10)
    km.fit(emb_train)
    d_tr = km.transform(emb_train)
    return np.hstack([emb_train, d_tr])
