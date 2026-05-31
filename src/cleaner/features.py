from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from cleaner.config import FEATURE_NAMES
from cleaner.utils import safe_age_years, safe_file_mb


def extract_basic_features(path: str) -> np.ndarray:
    """Return small handcrafted features that complement CLIP embeddings.

    These are cheap and explainable. They should not be used alone to delete photos;
    the personal classifier learns how much the user cares about each signal.
    """
    file_mb = safe_file_mb(path)
    age_years = safe_age_years(path)

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            width, height = im.size

            rgb = im.convert("RGB")
            small = ImageOps.contain(rgb, (256, 256))
            gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0

            brightness = float(gray.mean())
            contrast = float(gray.std())

            if gray.shape[0] > 1 and gray.shape[1] > 1:
                gy, gx = np.gradient(gray)
                sharp_raw = float(np.var(gx) + np.var(gy))
                sharpness = float(np.clip(np.log1p(sharp_raw * 1000.0) / 5.0, 0.0, 1.0))
            else:
                sharpness = 0.0
    except Exception:
        width, height = 0, 0
        brightness = 0.0
        contrast = 0.0
        sharpness = 0.0

    aspect_ratio = float(width / height) if height else 0.0

    name = str(path).lower()
    is_screenshot_name = 1.0 if any(
        token in name for token in ["screenshot", "screen shot", "스크린샷", "캡처"]
    ) else 0.0

    values = [
        file_mb,
        width / 1000.0,
        height / 1000.0,
        aspect_ratio,
        brightness,
        contrast,
        sharpness,
        age_years,
        is_screenshot_name,
    ]
    assert len(values) == len(FEATURE_NAMES)
    return np.asarray(values, dtype=np.float32)
