from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

FEATURE_NAMES = [
    "file_mb",
    "width_k",
    "height_k",
    "aspect_ratio",
    "brightness",
    "contrast",
    "sharpness",
    "age_years",
    "is_screenshot_name",
]

FEATURE_LABELS = {
    "file_mb": "file size",
    "width_k": "image width",
    "height_k": "image height",
    "aspect_ratio": "aspect ratio",
    "brightness": "brightness",
    "contrast": "contrast",
    "sharpness": "sharpness",
    "age_years": "photo age",
    "is_screenshot_name": "screenshot-like filename",
}

# XP thresholds by level. Level 1 starts at 0 XP.
LEVEL_THRESHOLDS = [0, 30, 80, 150, 250, 400, 650, 1000, 1500, 2200, 3200, 4600]


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = Path("cleaner_data")
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    fallback_pretrained: str = "openai"
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 16

    @property
    def labels_csv(self) -> Path:
        return self.data_dir / "labels.csv"

    @property
    def embedding_cache_file(self) -> Path:
        return self.data_dir / "embeddings.joblib"

    @property
    def model_file(self) -> Path:
        return self.data_dir / "personal_keep_delete_model.joblib"

    @property
    def stats_json(self) -> Path:
        return self.data_dir / "stats.json"

    @property
    def events_jsonl(self) -> Path:
        return self.data_dir / "events.jsonl"

    @property
    def delete_candidates_csv(self) -> Path:
        return self.data_dir / "delete_candidates.csv"


def get_config() -> AppConfig:
    data_dir = Path(os.getenv("CLEANER_DATA_DIR", "cleaner_data")).expanduser()
    device = os.getenv("CLEANER_DEVICE", "auto")
    model_name = os.getenv("CLEANER_MODEL_NAME", "ViT-B-32")
    pretrained = os.getenv("CLEANER_PRETRAINED", "laion2b_s34b_b79k")
    fallback_pretrained = os.getenv("CLEANER_FALLBACK_PRETRAINED", "openai")

    try:
        batch_size = int(os.getenv("CLEANER_BATCH_SIZE", "16"))
    except ValueError:
        batch_size = 16

    return AppConfig(
        data_dir=data_dir,
        model_name=model_name,
        pretrained=pretrained,
        fallback_pretrained=fallback_pretrained,
        device=device,
        batch_size=batch_size,
    )
