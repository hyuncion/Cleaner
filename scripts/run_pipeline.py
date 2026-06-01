from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cleaner.config import get_config
from cleaner.embeddings import update_embeddings
from cleaner.encoder import choose_device
from cleaner.labeling import import_label_csv
from cleaner.model import train_personal_model
from cleaner.recommendations import recommend_delete_candidates
from cleaner.storage import load_labels
from cleaner.utils import scan_photos


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal Cleaner AI pipeline.")
    parser.add_argument("--photo-dir", type=Path, default=Path("demo_photos"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--max-embeddings", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--include-labeled", action="store_true")
    parser.add_argument("--min-labels", type=int, default=20)
    args = parser.parse_args()

    cfg = get_config()
    if args.data_dir is not None:
        cfg = replace(cfg, data_dir=args.data_dir)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    paths = scan_photos(args.photo_dir)
    if not paths:
        raise SystemExit(f"No photos found in {args.photo_dir}")

    if args.labels_csv is not None:
        result = import_label_csv(cfg, args.labels_csv)
        print(f"imported labels: {result}")

    labels = load_labels(cfg)
    print(f"photos={len(paths)}")
    print(f"labels={len(labels)}")
    print(f"data_dir={cfg.data_dir.resolve()}")
    print(f"device={choose_device(cfg.device)}")

    def progress(done: int, total: int) -> None:
        print(f"embedding_progress={done}/{total}")

    processed = update_embeddings(
        cfg,
        paths,
        max_items=args.max_embeddings,
        progress_callback=progress,
    )
    print(f"new_embeddings={processed}")

    report = train_personal_model(cfg, min_labels=args.min_labels)
    print("train_report=" + json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("ok"):
        return

    df = recommend_delete_candidates(
        cfg,
        paths,
        top_n=args.top_n,
        only_unlabeled=not args.include_labeled,
    )
    if df.empty:
        print("No recommendations were produced.")
        return

    print(f"recommendations_csv={cfg.delete_candidates_csv.resolve()}")
    print(df.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
