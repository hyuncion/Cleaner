from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from cleaner.config import AppConfig
from cleaner.storage import upsert_label
from cleaner.utils import safe_file_mb, scan_photos

Action = Literal["keep", "discard", "important", "later"]

ACTION_TO_LABEL: dict[str, int] = {
    "keep": 1,
    "discard": 0,
    "important": 1,
    "later": -1,
}

ACTION_TO_WEIGHT: dict[str, float] = {
    "keep": 1.0,
    "discard": 1.0,
    "important": 2.0,
    "later": 0.0,
}


def record_action(
    cfg: AppConfig,
    path: str | Path,
    action: Action,
    source: str = "manual",
    reason: str = "",
) -> None:
    if action not in ACTION_TO_LABEL:
        raise ValueError(f"Unknown action: {action}")

    upsert_label(
        cfg,
        path=str(path),
        label=ACTION_TO_LABEL[action],
        action=action,
        source=source,
        reason=reason,
        weight=ACTION_TO_WEIGHT[action],
    )


def export_label_template(
    cfg: AppConfig,
    photo_dir: str | Path,
    out_csv: str | Path | None = None,
    max_items: int = 200,
) -> Path:
    paths = scan_photos(photo_dir)
    if max_items > 0:
        paths = paths[:max_items]

    out_path = Path(out_csv) if out_csv else cfg.data_dir / "label_template.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "path": paths,
            "label": "",
            "action": "",
            "notes": "",
            "file_mb": [safe_file_mb(path) for path in paths],
        }
    )
    df.to_csv(out_path, index=False)
    return out_path


def import_label_csv(
    cfg: AppConfig,
    csv_path: str | Path,
    source: str = "csv_import",
) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("CSV must contain a `path` column.")
    if "label" not in df.columns and "action" not in df.columns:
        raise ValueError("CSV must contain `label` or `action`.")

    imported = 0
    skipped = 0
    for _, row in df.iterrows():
        path = str(row["path"])
        action = str(row.get("action", "")).strip().lower()
        label_value = row.get("label")

        if action in ACTION_TO_LABEL:
            record_action(cfg, path, action=action, source=source, reason=str(row.get("notes", "")))
            imported += 1
            continue

        try:
            label = int(float(label_value))
        except Exception:
            skipped += 1
            continue

        if label not in {0, 1, -1}:
            skipped += 1
            continue

        upsert_label(
            cfg,
            path=path,
            label=label,
            action="keep" if label == 1 else "discard" if label == 0 else "later",
            source=source,
            reason=str(row.get("notes", "")),
            weight=1.0 if label in {0, 1} else 0.0,
        )
        imported += 1

    return {"imported": imported, "skipped": skipped}
