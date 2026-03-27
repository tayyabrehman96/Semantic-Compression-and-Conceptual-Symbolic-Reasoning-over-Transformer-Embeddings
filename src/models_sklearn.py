"""Sklearn classifiers used on top of frozen embeddings (matches notebook-style bank)."""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from . import config


def classifier_bank(num_classes: int, random_state: int | None = None) -> dict:
    """Return name -> unfitted estimator. XGBoost included if package is installed."""
    rs = random_state if random_state is not None else config.RANDOM_STATE
    bank: dict = {
        "Linear SVM": SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=rs,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=500,
            n_jobs=-1,
            class_weight="balanced",
            random_state=rs,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=50,
            max_depth=20,
            n_jobs=-1,
            class_weight="balanced",
            random_state=rs,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=rs,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=rs,
        ),
    }
    try:
        from xgboost import XGBClassifier

        xgb_kw = dict(
            objective="multi:softprob",
            num_class=num_classes,
            n_estimators=50,
            max_depth=5,
            eval_metric="mlogloss",
            random_state=rs,
        )
        try:
            bank["XGBoost"] = XGBClassifier(**xgb_kw, use_label_encoder=False)
        except TypeError:
            bank["XGBoost"] = XGBClassifier(**xgb_kw)
    except ImportError:
        pass
    return bank
