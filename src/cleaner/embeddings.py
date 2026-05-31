from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from cleaner.config import AppConfig
from cleaner.encoder import load_clip_encoder
from cleaner.features import extract_basic_features
from cleaner.storage import (
    cache_entry_is_valid,
    load_embedding_cache,
    log_event,
    save_embedding_cache,
)

ProgressCallback = Callable[[int, int], None]


def update_embeddings(
    cfg: AppConfig,
    paths: list[str],
    max_items: int = 300,
    progress_callback: ProgressCallback | None = None,
) -> int:
    """Create/update embedding cache for photos.

    Returns the number of newly processed images.
    """
    encoder = load_clip_encoder(cfg)
    cache = load_embedding_cache(cfg)

    todo = [p for p in paths if not cache_entry_is_valid(p, cache.get(p))]
    if max_items > 0:
        todo = todo[:max_items]

    if not todo:
        return 0

    done = 0
    batch_size = max(int(cfg.batch_size), 1)

    for start in range(0, len(todo), batch_size):
        batch_paths = todo[start : start + batch_size]
        embeddings, valid_paths = encoder.encode_images(batch_paths)

        for path, vec in zip(valid_paths, embeddings):
            try:
                stat = Path(path).stat()
                cache[path] = {
                    "embedding": vec,
                    "features": extract_basic_features(path),
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
                done += 1
            except Exception:
                continue

        if progress_callback is not None:
            progress_callback(done, len(todo))

        if done and done % 128 == 0:
            save_embedding_cache(cfg, cache)

    save_embedding_cache(cfg, cache)
    log_event(cfg, "embeddings_updated", {"count": done})
    return done


def make_feature_matrix(
    cfg: AppConfig,
    paths: list[str],
) -> tuple[np.ndarray, list[str]]:
    cache = load_embedding_cache(cfg)
    rows: list[np.ndarray] = []
    used_paths: list[str] = []

    for path in paths:
        entry = cache.get(path)
        if not entry:
            continue

        emb = entry.get("embedding")
        basic = entry.get("features")
        if emb is None or basic is None:
            continue

        row = np.concatenate(
            [np.asarray(emb, dtype=np.float32), np.asarray(basic, dtype=np.float32)],
            axis=0,
        )
        rows.append(row)
        used_paths.append(path)

    if not rows:
        return np.empty((0, 0), dtype=np.float32), []

    return np.vstack(rows).astype(np.float32), used_paths
