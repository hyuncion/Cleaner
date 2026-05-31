from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cleaner.config import get_config
from cleaner.storage import upsert_label
from cleaner.utils import scan_photos


def infer_demo_label(path: str) -> tuple[int, str]:
    name = Path(path).name.lower()
    if "screenshot" in name or "스크린샷" in name or "capture" in name or "캡처" in name:
        return 0, "demo_screenshot"
    if "blurry" in name or "blur" in name:
        return 0, "demo_blurry"
    return 1, "demo_keep"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic demo keep/delete labels.")
    parser.add_argument("--photo-dir", type=Path, default=Path("demo_photos"))
    parser.add_argument("--max", type=int, default=80)
    args = parser.parse_args()

    cfg = get_config()
    paths = scan_photos(args.photo_dir)[: args.max]

    if not paths:
        raise SystemExit(f"No photos found in {args.photo_dir}")

    keep = 0
    discard = 0
    for path in paths:
        label, reason = infer_demo_label(path)
        upsert_label(cfg, path, label=label, source="demo_auto_label", reason=reason, weight=1.0)
        if label == 1:
            keep += 1
        else:
            discard += 1

    print(f"Created labels for {len(paths)} photos: keep={keep}, discard={discard}")
    print(f"labels.csv: {cfg.labels_csv.resolve()}")


if __name__ == "__main__":
    main()
