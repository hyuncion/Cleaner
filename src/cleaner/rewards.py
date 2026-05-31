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
        badges.append("첫 정리 시작")
    if total_labeled >= 100:
        badges.append("100장 판단")
    if total_labeled >= 500:
        badges.append("500장 정리러")
    if discard_mb >= 1024:
        badges.append("1GB 세이버")
    if train_count >= 1:
        badges.append("AI 첫 학습")
    if train_count >= 5:
        badges.append("AI 트레이너")
    return badges
