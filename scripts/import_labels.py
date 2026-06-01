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
from cleaner.labeling import import_label_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Import labels from a CSV file.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config()
    if args.data_dir is not None:
        cfg = replace(cfg, data_dir=args.data_dir)

    result = import_label_csv(cfg, args.csv_path)
    print(f"imported={result['imported']} skipped={result['skipped']}")
    print(f"labels_csv={cfg.labels_csv.resolve()}")


if __name__ == "__main__":
    main()
