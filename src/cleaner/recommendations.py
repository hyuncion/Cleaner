from __future__ import annotations

import random

import pandas as pd

from cleaner.config import AppConfig
from cleaner.embeddings import make_feature_matrix
from cleaner.model import get_model_bundle, predict_keep_probability
from cleaner.storage import load_labels
from cleaner.utils import safe_file_mb


def recommend_delete_candidates(
    cfg: AppConfig,
    all_paths: list[str],
    top_n: int = 100,
    only_unlabeled: bool = True,
) -> pd.DataFrame:
    bundle = get_model_bundle(cfg)
    if not bundle:
        return pd.DataFrame()

    model = bundle["model"]
    labels = load_labels(cfg)
    labeled_set = set(labels["path"].tolist())

    candidate_paths = [p for p in all_paths if p not in labeled_set] if only_unlabeled else all_paths
    x, used_paths = make_feature_matrix(cfg, candidate_paths)

    if x.shape[0] == 0:
        return pd.DataFrame()

    p_keep = predict_keep_probability(model, x)
    p_delete = 1.0 - p_keep

    df = pd.DataFrame({"path": used_paths, "p_delete": p_delete, "p_keep": p_keep})
    df["file_mb"] = df["path"].map(safe_file_mb)
    df = df.sort_values("p_delete", ascending=False).head(top_n).reset_index(drop=True)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.delete_candidates_csv, index=False)
    return df


def pick_next_photo(cfg: AppConfig, paths: list[str], mode: str) -> str | None:
    labels = load_labels(cfg)
    labeled_set = set(labels["path"].tolist())
    unlabeled = [p for p in paths if p not in labeled_set]

    if not unlabeled:
        return None

    if mode == "랜덤":
        return random.choice(unlabeled)

    if mode == "AI가 헷갈리는 사진":
        df = recommend_delete_candidates(cfg, paths, top_n=1000, only_unlabeled=True)
        if not df.empty:
            df["uncertainty"] = (df["p_delete"] - 0.5).abs()
            return str(df.sort_values("uncertainty", ascending=True).iloc[0]["path"])

    return unlabeled[0]
