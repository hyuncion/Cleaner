from __future__ import annotations

import streamlit as st

from cleaner.config import AppConfig
from cleaner.embeddings import update_embeddings
from cleaner.model import train_personal_model
from cleaner.rewards import level_from_xp, xp_progress
from cleaner.storage import load_labels, load_stats
from cleaner.utils import safe_file_mb


def render_sidebar(paths: list[str], cfg: AppConfig) -> None:
    labels = load_labels(cfg)
    stats = load_stats(cfg)

    keep_count = int((labels["label"] == 1).sum())
    discard_count = int((labels["label"] == 0).sum())
    total_labeled = keep_count + discard_count

    discard_mb = 0.0
    if discard_count:
        discard_mb = sum(safe_file_mb(p) for p in labels[labels["label"] == 0]["path"].tolist())

    xp = int(stats.get("xp", 0))
    level = level_from_xp(xp)
    _, next_xp, progress = xp_progress(xp)

    st.sidebar.subheader("상태")
    st.sidebar.metric("찾은 사진", f"{len(paths):,}장")
    st.sidebar.metric("판단 완료", f"{total_labeled:,}장")
    st.sidebar.metric("남김 / 버림", f"{keep_count:,} / {discard_count:,}")
    st.sidebar.metric("확보 예정 용량", f"{discard_mb:,.1f} MB")
    st.sidebar.metric("AI 레벨", f"Lv.{level}")
    st.sidebar.progress(progress)
    st.sidebar.caption(f"XP {xp} / 다음 레벨 {next_xp}")

    st.sidebar.divider()

    max_items = st.sidebar.number_input(
        "이번에 임베딩 만들 사진 수",
        min_value=10,
        max_value=10000,
        value=300,
        step=50,
    )

    if st.sidebar.button("임베딩 생성/업데이트"):
        progress_bar = st.sidebar.progress(0.0)
        status_box = st.sidebar.empty()

        def progress(done: int, total: int) -> None:
            progress_bar.progress(min(done / max(total, 1), 1.0))
            status_box.write(f"임베딩 생성 중: {done} / {total}")

        with st.spinner("pretrained image encoder를 불러오는 중입니다. 첫 실행은 시간이 걸릴 수 있습니다."):
            try:
                count = update_embeddings(cfg, paths, max_items=int(max_items), progress_callback=progress)
            except RuntimeError as exc:
                st.sidebar.error(str(exc))
                return

        if count:
            st.sidebar.success(f"{count:,}장 임베딩 완료")
        else:
            st.sidebar.info("새로 만들 임베딩이 없습니다.")

    if st.sidebar.button("내 AI 학습시키기"):
        with st.spinner("개인화 keep/delete 모델 학습 중..."):
            report = train_personal_model(cfg)

        if report.get("ok"):
            st.sidebar.success("학습 완료")
        else:
            st.sidebar.warning(report.get("message", "학습 실패"))
