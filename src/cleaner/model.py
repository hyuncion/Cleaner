from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cleaner.config import AppConfig, FEATURE_LABELS, FEATURE_NAMES
from cleaner.embeddings import make_feature_matrix
from cleaner.storage import (
    load_labels,
    load_model_bundle,
    load_stats,
    log_event,
    save_model_bundle,
    save_stats,
)
from cleaner.utils import now_iso


def make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )


def predict_keep_probability(model: Pipeline, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.named_steps["clf"].classes_)
    if 1 in classes:
        keep_index = classes.index(1)
        return probabilities[:, keep_index]
    return np.zeros(x.shape[0], dtype=np.float32)


def learned_feature_summary(model: Pipeline) -> dict[str, list[dict[str, Any]]]:
    """Summarize only handcrafted feature coefficients for basic interpretability."""
    try:
        clf = model.named_steps["clf"]
        coefficients = clf.coef_[0]
        basic_coefficients = coefficients[-len(FEATURE_NAMES) :]
        pairs = list(zip(FEATURE_NAMES, basic_coefficients))
        pairs_sorted = sorted(pairs, key=lambda item: item[1], reverse=True)

        keep_signals = [
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "direction": "keep",
                "coef": float(value),
            }
            for name, value in pairs_sorted[:3]
        ]
        discard_signals = [
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "direction": "discard",
                "coef": float(value),
            }
            for name, value in pairs_sorted[-3:]
        ]
        return {"keep_signals": keep_signals, "discard_signals": discard_signals}
    except Exception:
        return {"keep_signals": [], "discard_signals": []}


def _training_labels(cfg: AppConfig) -> pd.DataFrame:
    labels = load_labels(cfg)
    labels = labels[labels["label"].isin([0, 1])].copy()
    labels["label"] = labels["label"].astype(int)
    return labels


def train_personal_model(cfg: AppConfig, min_labels: int = 20) -> dict[str, Any]:
    labels = _training_labels(cfg)

    if len(labels) < min_labels:
        return {
            "ok": False,
            "message": f"Need at least {min_labels} keep/discard labels before training.",
            "labels_used": int(len(labels)),
        }

    if labels["label"].nunique() < 2:
        return {
            "ok": False,
            "message": "Need both keep=1 and discard=0 labels before training.",
            "labels_used": int(len(labels)),
        }

    labeled_paths = labels["path"].tolist()
    x, used_paths = make_feature_matrix(cfg, labeled_paths)

    if len(used_paths) < min_labels:
        return {
            "ok": False,
            "message": "Not enough cached embeddings. Run embedding update first.",
            "labels_used": int(len(used_paths)),
        }

    used_labels = labels.set_index("path").loc[used_paths].reset_index()
    y = used_labels["label"].astype(int).to_numpy()
    weights = used_labels["weight"].fillna(1.0).astype(float).to_numpy()

    keep_count = int((y == 1).sum())
    discard_count = int((y == 0).sum())
    if keep_count == 0 or discard_count == 0:
        return {"ok": False, "message": "Need both keep and discard examples."}

    report: dict[str, Any] = {
        "ok": True,
        "trained_at": now_iso(),
        "labels_used": int(len(y)),
        "keep_count": keep_count,
        "discard_count": discard_count,
        "accuracy": None,
        "auc": None,
        "message": "Training complete.",
    }

    min_class_count = min(keep_count, discard_count)
    if len(y) >= 40 and min_class_count >= 8:
        x_train, x_test, y_train, y_test, w_train, _ = train_test_split(
            x,
            y,
            weights,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        tmp_model = make_classifier()
        tmp_model.fit(x_train, y_train, clf__sample_weight=w_train)
        p_keep = predict_keep_probability(tmp_model, x_test)
        pred = (p_keep >= 0.5).astype(int)
        report["accuracy"] = float(accuracy_score(y_test, pred))
        if len(np.unique(y_test)) == 2:
            report["auc"] = float(roc_auc_score(y_test, p_keep))

    final_model = make_classifier()
    final_model.fit(x, y, clf__sample_weight=weights)
    report["learned_summary"] = learned_feature_summary(final_model)

    save_model_bundle(
        cfg,
        {
            "model": final_model,
            "feature_names": FEATURE_NAMES,
            "trained_at": report["trained_at"],
            "report": report,
        },
    )

    stats = load_stats(cfg)
    stats["train_count"] = int(stats.get("train_count", 0)) + 1
    stats["xp"] = int(stats.get("xp", 0)) + int(30 + len(y) * 0.2)
    stats["last_train_report"] = report
    save_stats(cfg, stats)

    log_event(cfg, "model_trained", report)
    return report


def get_model_bundle(cfg: AppConfig) -> dict[str, Any] | None:
    return load_model_bundle(cfg)
