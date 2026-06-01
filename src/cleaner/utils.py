from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from cleaner.config import IMAGE_EXTS


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_file_mb(path: str | Path) -> float:
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def safe_age_years(path: str | Path) -> float:
    try:
        modified_at = Path(path).stat().st_mtime
        return max((time.time() - modified_at) / (365.25 * 24 * 3600), 0.0)
    except Exception:
        return 0.0


def scan_photos(photo_dir: str | Path) -> list[str]:
    root = Path(photo_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    paths = [
        str(path.resolve())
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(paths)


def is_colab_runtime() -> bool:
    return bool(os.getenv("COLAB_RELEASE_TAG")) or Path("/content").exists()


def default_photo_dir() -> str:
    env_value = os.getenv("CLEANER_DEFAULT_PHOTO_DIR")
    if env_value:
        return str(Path(env_value).expanduser())

    candidates = [
        Path.cwd() / "demo_photos",
        Path("/content/Cleaner/demo_photos"),
        Path("/content/demo_photos"),
        Path.home() / "Pictures",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return str(candidate)

    return str(Path.cwd() / "demo_photos")
