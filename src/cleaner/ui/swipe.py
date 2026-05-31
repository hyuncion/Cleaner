from __future__ import annotations

from pathlib import Path

import streamlit as st

from cleaner.config import AppConfig
from cleaner.recommendations import pick_next_photo
from cleaner.storage import add_xp, load_labels, upsert_label


def render_swipe_tab(paths: list[str], cfg: AppConfig) -> None:
    st.subheader("스와이프식 정리")

    mode = st.radio("정리 모드", ["기본", "랜덤", "AI가 헷갈리는 사진"], horizontal=True)

    labels = load_labels(cfg)
    labeled_count = len(labels)
    next_train = 100 - (labeled_count % 100) if labeled_count else 100

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("남은 후보", f"{max(len(paths) - labeled_count, 0):,}장")
    col_b.metric("다음 학습 배치까지", f"{next_train}장")
    col_c.metric("라벨 데이터", f"{labeled_count:,}개")

    if "current_photo" not in st.session_state:
        st.session_state.current_photo = None

    if st.session_state.current_photo is None:
        st.session_state.current_photo = pick_next_photo(cfg, paths, mode)

    current = st.session_state.current_photo
    if current is None:
        st.success("라벨링할 사진이 없습니다.")
        return

    photo_path = Path(current)
    if not photo_path.exists():
        st.warning("사진 파일을 찾을 수 없습니다. 다음 사진으로 넘어갑니다.")
        st.session_state.current_photo = None
        st.rerun()

    st.caption(str(photo_path))

    try:
        st.image(current, use_container_width=True)
    except Exception:
        st.error("이 이미지는 표시할 수 없습니다.")
        if st.button("다음 사진"):
            st.session_state.current_photo = None
            st.rerun()
        return

    with st.expander("버리는 이유를 기록하고 싶다면 선택"):
        reason = st.selectbox(
            "reason",
            [
                "",
                "duplicate",
                "blurry",
                "bad_expression",
                "screenshot",
                "old_info",
                "not_needed",
                "sensitive",
                "other",
            ],
        )

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("⬅️ 버림 0", use_container_width=True):
        upsert_label(cfg, current, label=0, source="manual_swipe", reason=reason, weight=1.0)
        add_xp(cfg, 1)
        st.session_state.current_photo = None
        st.rerun()

    if col2.button("➡️ 남김 1", use_container_width=True):
        upsert_label(cfg, current, label=1, source="manual_swipe", reason="", weight=1.0)
        add_xp(cfg, 1)
        st.session_state.current_photo = None
        st.rerun()

    if col3.button("⭐ 중요하게 남김", use_container_width=True):
        upsert_label(cfg, current, label=1, source="manual_favorite", reason="favorite", weight=2.0)
        add_xp(cfg, 3)
        st.session_state.current_photo = None
        st.rerun()

    if col4.button("⏭️ 나중에", use_container_width=True):
        st.session_state.current_photo = None
        st.rerun()
