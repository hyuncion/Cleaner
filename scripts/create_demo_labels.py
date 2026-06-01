from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cleaner.config import get_config
from cleaner.labeling import record_action
from cleaner.utils import scan_photos


def infer_demo_action(path: str) -> tuple[str, str]:
    name = Path(path).name.lower()
    if "screenshot" in name or "capture" in name:
        return "discard", "demo_screenshot"
    if "blurry" in name or "blur" in name:
        return "discard", "demo_blurry"
    return "keep", "demo_keep"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic demo keep/discard labels.")
    parser.add_argument("--photo-dir", type=Path, default=Path("demo_photos"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--max", type=int, default=80)
    args = parser.parse_args()

    cfg = get_config()
    if args.data_dir is not None:
        cfg = replace(cfg, data_dir=args.data_dir)

    paths = scan_photos(args.photo_dir)[: args.max]
    if not paths:
        raise SystemExit(f"No photos found in {args.photo_dir}")

    keep = 0
    discard = 0
    for path in paths:
        action, reason = infer_demo_action(path)
        record_action(cfg, path, action=action, source="demo_auto_label", reason=reason)
        if action == "keep":
            keep += 1
        elif action == "discard":
            discard += 1

    print(f"Created labels for {len(paths)} photos: keep={keep}, discard={discard}")
    print(f"labels_csv={cfg.labels_csv.resolve()}")


if __name__ == "__main__":
    main()
