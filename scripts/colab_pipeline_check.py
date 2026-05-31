from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cleaner.config import get_config
from cleaner.embeddings import update_embeddings
from cleaner.model import train_personal_model
from cleaner.recommendations import recommend_delete_candidates
from cleaner.storage import load_labels
from cleaner.utils import scan_photos


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small Cleaner pipeline check in Colab/local.")
    parser.add_argument("--photo-dir", type=Path, default=Path("demo_photos"))
    parser.add_argument("--max-embeddings", type=int, default=40)
    parser.add_argument("--recommend", type=int, default=10)
    args = parser.parse_args()

    cfg = get_config()
    paths = scan_photos(args.photo_dir)
    print(f"photos: {len(paths)}")
    print(f"data_dir: {cfg.data_dir.resolve()}")
    print(f"device request: {cfg.device}")

    if not paths:
        raise SystemExit(f"No photos found in {args.photo_dir}")

    def progress(done: int, total: int) -> None:
        print(f"embedding progress: {done}/{total}")

    processed = update_embeddings(
        cfg,
        paths,
        max_items=args.max_embeddings,
        progress_callback=progress,
    )
    print(f"new embeddings: {processed}")

    labels = load_labels(cfg)
    print(f"labels: {len(labels)}")

    report = train_personal_model(cfg)
    print("train report:", report)

    if report.get("ok"):
        df = recommend_delete_candidates(cfg, paths, top_n=args.recommend, only_unlabeled=False)
        print(df.head(args.recommend).to_string(index=False))


if __name__ == "__main__":
    main()
