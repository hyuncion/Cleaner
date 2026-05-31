from __future__ import annotations

import streamlit as st

from cleaner.config import AppConfig
from cleaner.rewards import badges_for, level_from_xp, xp_progress
from cleaner.storage import load_labels, load_stats
from cleaner.utils import safe_file_mb


def render_rewards_tab(cfg: AppConfig) -> None:
    st.subheader("보상 / 리포트")

    labels = load_labels(cfg)
    stats = load_stats(cfg)

    xp = int(stats.get("xp", 0))
    level = level_from_xp(xp)
    _, next_xp, progress = xp_progress(xp)

    keep_count = int((labels["label"] == 1).sum())
    discard_count = int((labels["label"] == 0).sum())
    total_labeled = keep_count + discard_count
    discard_mb = sum(safe_file_mb(p) for p in labels[labels["label"] == 0]["path"].tolist())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AI 레벨", f"Lv.{level}")
    col2.metric("XP", f"{xp:,}")
    col3.metric("판단한 사진", f"{total_labeled:,}장")
    col4.metric("확보 예정", f"{discard_mb:,.1f} MB")

    st.progress(progress)
    st.caption(f"다음 레벨까지 {max(next_xp - xp, 0)} XP")

    st.divider()
    st.write("획득한 배지")

    badges = badges_for(total_labeled, discard_mb, int(stats.get("train_count", 0)))
    if badges:
        st.write(" · ".join([f"🏅 {badge}" for badge in badges]))
    else:
        st.info("아직 배지가 없습니다. 10장만 판단해보세요.")

    st.divider()
    st.write("AI가 최근 학습한 것")

    last_report = stats.get("last_train_report")
    if not last_report:
        st.info("아직 학습 리포트가 없습니다.")
        return

    st.json(
        {
            "trained_at": last_report.get("trained_at"),
            "labels_used": last_report.get("labels_used"),
            "keep_count": last_report.get("keep_count"),
            "discard_count": last_report.get("discard_count"),
            "accuracy": last_report.get("accuracy"),
            "auc": last_report.get("auc"),
        }
    )

    summary = last_report.get("learned_summary", {})
    keep_signals = summary.get("keep_signals", [])
    discard_signals = summary.get("discard_signals", [])

    if keep_signals:
        st.write("남김 쪽으로 작용한 신호")
        for item in keep_signals:
            st.write(f"- {item['label']}")

    if discard_signals:
        st.write("버림 쪽으로 작용한 신호")
        for item in discard_signals:
            st.write(f"- {item['label']}")
