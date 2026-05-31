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

FEATURE_KO = {
    "file_mb": "파일 크기",
    "width_k": "이미지 너비",
    "height_k": "이미지 높이",
    "aspect_ratio": "가로세로 비율",
    "brightness": "밝기",
    "contrast": "대비",
    "sharpness": "선명도",
    "age_years": "오래된 정도",
    "is_screenshot_name": "스크린샷 이름 패턴",
}

# XP thresholds by level. Lv.1 starts at 0 XP.
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
    try:
        batch_size = int(os.getenv("CLEANER_BATCH_SIZE", "16"))
    except ValueError:
        batch_size = 16

    return AppConfig(
        data_dir=data_dir,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        batch_size=batch_size,
    )
