from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps

from cleaner.config import AppConfig


@dataclass
class ClipEncoder:
    model: torch.nn.Module
    preprocess: Callable
    device: str

    def encode_images(self, paths: list[str]) -> tuple[np.ndarray, list[str]]:
        images = []
        valid_paths: list[str] = []

        for path in paths:
            try:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    images.append(self.preprocess(im))
                    valid_paths.append(path)
            except Exception:
                continue

        if not images:
            return np.empty((0, 0), dtype=np.float32), []

        image_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(image_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.detach().cpu().numpy().astype(np.float32), valid_paths


def choose_device(requested: str = "auto") -> str:
    requested = requested.lower()
    if requested in {"cpu", "cuda", "mps"}:
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip_encoder(cfg: AppConfig) -> ClipEncoder:
    """Load OpenCLIP encoder.

    Import open_clip lazily so modules/tests that do not need embedding can run
    without immediately loading large model dependencies.
    """
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open-clip-torch가 설치되어 있지 않습니다. `pip install open-clip-torch`를 실행하세요."
        ) from exc

    device = choose_device(cfg.device)

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.model_name,
            pretrained=cfg.pretrained,
        )
    except Exception:
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.model_name,
            pretrained=cfg.fallback_pretrained,
        )

    model.eval()
    model.to(device)
    return ClipEncoder(model=model, preprocess=preprocess, device=device)
