from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run app.py` from a fresh clone without requiring `pip install -e .` first.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from cleaner.config import get_config
from cleaner.ui.recommendations import render_recommendations_tab
from cleaner.ui.rewards import render_rewards_tab
from cleaner.ui.sidebar import render_sidebar
from cleaner.ui.swipe import render_swipe_tab
from cleaner.utils import default_photo_dir, is_colab_runtime, scan_photos


def main() -> None:
    st.set_page_config(page_title="Cleaner MVP", page_icon="🧹", layout="wide")

    cfg = get_config()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    st.title("🧹 Cleaner MVP")
    st.caption("스와이프할수록 똑똑해지는 개인 사진 정리 AI")

    if is_colab_runtime():
        st.info(
            "Colab 환경으로 감지되었습니다. 데모 사진을 만들었다면 기본 경로는 "
            "`/content/Cleaner/demo_photos` 또는 `demo_photos`입니다."
        )

    if "photo_dir" not in st.session_state:
        st.session_state.photo_dir = default_photo_dir()

    photo_dir = st.text_input(
        "사진 폴더 경로",
        value=st.session_state.photo_dir,
        help=(
            "예: /Users/you/Pictures/exported_photos, "
            "C:\\Users\\you\\Pictures, /content/Cleaner/demo_photos"
        ),
    )

    if photo_dir != st.session_state.photo_dir:
        st.session_state.photo_dir = photo_dir
        st.session_state.current_photo = None
        st.session_state.recommend_df = None
        st.session_state.paths = scan_photos(photo_dir)

    if "paths" not in st.session_state:
        st.session_state.paths = scan_photos(photo_dir)

    rescan_col, _ = st.columns([1, 4])
    if rescan_col.button("다시 스캔"):
        st.session_state.paths = scan_photos(photo_dir)
        st.session_state.current_photo = None
        st.session_state.recommend_df = None
        st.rerun()

    paths = st.session_state.paths

    if not paths:
        st.warning("사진을 찾지 못했습니다. 폴더 경로를 확인하거나 jpg/png/webp 사진이 있는 폴더를 지정하세요.")
        return

    render_sidebar(paths, cfg)

    tab1, tab2, tab3 = st.tabs(["1. 스와이프 정리", "2. AI 추천 검수", "3. 보상 / 리포트"])

    with tab1:
        render_swipe_tab(paths, cfg)

    with tab2:
        render_recommendations_tab(paths, cfg)

    with tab3:
        render_rewards_tab(cfg)


if __name__ == "__main__":
    main()
