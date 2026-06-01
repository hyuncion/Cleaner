from __future__ import annotations

from cleaner.config import LEVEL_THRESHOLDS


def level_from_xp(xp: int) -> int:
    level = 1
    for threshold in LEVEL_THRESHOLDS[1:]:
        if xp >= threshold:
            level += 1
    return level


def xp_progress(xp: int) -> tuple[int, int, float]:
    level = level_from_xp(xp)
    current_threshold = LEVEL_THRESHOLDS[min(level - 1, len(LEVEL_THRESHOLDS) - 1)]

    if level < len(LEVEL_THRESHOLDS):
        next_threshold = LEVEL_THRESHOLDS[level]
    else:
        next_threshold = current_threshold + 2000

    progress = (xp - current_threshold) / max(next_threshold - current_threshold, 1)
    return current_threshold, next_threshold, max(0.0, min(float(progress), 1.0))


def badges_for(total_labeled: int, discard_mb: float, train_count: int) -> list[str]:
    badges: list[str] = []
    if total_labeled >= 10:
        badges.append("First cleanup")
    if total_labeled >= 100:
        badges.append("100 decisions")
    if total_labeled >= 500:
        badges.append("500 decisions")
    if discard_mb >= 1024:
        badges.append("1GB saved")
    if train_count >= 1:
        badges.append("First AI training")
    if train_count >= 5:
        badges.append("AI trainer")
    return badges
