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
from cleaner.labeling import export_label_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CSV template for manual labeling.")
    parser.add_argument("--photo-dir", type=Path, default=Path("demo_photos"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max", type=int, default=200)
    args = parser.parse_args()

    cfg = get_config()
    if args.data_dir is not None:
        cfg = replace(cfg, data_dir=args.data_dir)

    out_path = export_label_template(
        cfg=cfg,
        photo_dir=args.photo_dir,
        out_csv=args.out,
        max_items=args.max,
    )
    print(f"label_template_csv={out_path.resolve()}")
    print("Fill `label` with 1=keep, 0=discard, -1=later, then import it.")


if __name__ == "__main__":
    main()
