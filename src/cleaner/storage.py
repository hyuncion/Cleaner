from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from cleaner.config import AppConfig
from cleaner.utils import now_iso

LABEL_COLUMNS = ["path", "label", "source", "reason", "weight", "created_at"]


def ensure_data_dir(cfg: AppConfig) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {**default, **data}
    except Exception:
        return dict(default)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_event(cfg: AppConfig, event_type: str, payload: dict[str, Any]) -> None:
    ensure_data_dir(cfg)
    event = {"ts": now_iso(), "event_type": event_type, "payload": payload}
    with cfg.events_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_labels(cfg: AppConfig) -> pd.DataFrame:
    ensure_data_dir(cfg)
    if not cfg.labels_csv.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)

    df = pd.read_csv(cfg.labels_csv)
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[LABEL_COLUMNS]
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    return df


def save_labels(cfg: AppConfig, df: pd.DataFrame) -> None:
    ensure_data_dir(cfg)
    df.to_csv(cfg.labels_csv, index=False)


def upsert_label(
    cfg: AppConfig,
    path: str,
    label: int,
    source: str,
    reason: str = "",
    weight: float = 1.0,
) -> None:
    df = load_labels(cfg)

    # If the same photo is labeled again, keep the latest decision.
    df = df[df["path"] != path]

    new_row = {
        "path": path,
        "label": int(label),
        "source": source,
        "reason": reason,
        "weight": float(weight),
        "created_at": now_iso(),
    }
    if df.empty:
        df = pd.DataFrame([new_row], columns=LABEL_COLUMNS)
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_labels(cfg, df)

    log_event(
        cfg,
        "photo_labeled",
        {"path": path, "label": int(label), "source": source, "reason": reason, "weight": weight},
    )


def load_stats(cfg: AppConfig) -> dict[str, Any]:
    default = {"xp": 0, "train_count": 0, "last_train_report": None}
    return read_json(cfg.stats_json, default)


def save_stats(cfg: AppConfig, stats: dict[str, Any]) -> None:
    write_json(cfg.stats_json, stats)


def add_xp(cfg: AppConfig, amount: int) -> None:
    stats = load_stats(cfg)
    stats["xp"] = int(stats.get("xp", 0)) + int(amount)
    save_stats(cfg, stats)


def load_embedding_cache(cfg: AppConfig) -> dict[str, dict[str, Any]]:
    ensure_data_dir(cfg)
    if not cfg.embedding_cache_file.exists():
        return {}
    try:
        return joblib.load(cfg.embedding_cache_file)
    except Exception:
        return {}


def save_embedding_cache(cfg: AppConfig, cache: dict[str, dict[str, Any]]) -> None:
    ensure_data_dir(cfg)
    joblib.dump(cache, cfg.embedding_cache_file)


def cache_entry_is_valid(path: str, entry: dict[str, Any] | None) -> bool:
    if not entry:
        return False
    try:
        stat = Path(path).stat()
    except Exception:
        return False

    return (
        entry.get("mtime") == stat.st_mtime
        and entry.get("size") == stat.st_size
        and "embedding" in entry
        and "features" in entry
    )


def save_model_bundle(cfg: AppConfig, bundle: dict[str, Any]) -> None:
    ensure_data_dir(cfg)
    joblib.dump(bundle, cfg.model_file)


def load_model_bundle(cfg: AppConfig) -> dict[str, Any] | None:
    if not cfg.model_file.exists():
        return None
    try:
        return joblib.load(cfg.model_file)
    except Exception:
        return None
