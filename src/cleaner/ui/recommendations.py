from __future__ import annotations

import streamlit as st

from cleaner.config import AppConfig
from cleaner.model import get_model_bundle
from cleaner.recommendations import recommend_delete_candidates
from cleaner.storage import add_xp, upsert_label


def render_recommendations_tab(paths: list[str], cfg: AppConfig) -> None:
    st.subheader("AI 삭제 후보 검수")

    bundle = get_model_bundle(cfg)
    if not bundle:
        st.info("아직 학습된 모델이 없습니다. 라벨링 후 왼쪽 메뉴에서 '내 AI 학습시키기'를 눌러주세요.")
        return

    report = bundle.get("report", {})
    st.caption(f"마지막 학습: {bundle.get('trained_at', 'unknown')}")

    if report.get("accuracy") is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("학습 데이터", f"{report.get('labels_used', 0):,}장")
        col2.metric("검증 Accuracy", f"{report.get('accuracy', 0):.2f}")
        if report.get("auc") is not None:
            col3.metric("검증 AUC", f"{report.get('auc', 0):.2f}")
        else:
            col3.metric("검증 AUC", "-")
    else:
        st.warning("데이터가 아직 적어서 검증 점수는 표시하지 않습니다.")

    top_n = st.slider("추천 후보 수", 10, 300, 50, 10)

    if st.button("삭제 후보 추천받기"):
        with st.spinner("AI가 삭제 후보를 고르는 중..."):
            st.session_state.recommend_df = recommend_delete_candidates(
                cfg,
                paths,
                top_n=top_n,
                only_unlabeled=True,
            )

    df = st.session_state.get("recommend_df")
    if df is None or df.empty:
        st.info("버튼을 눌러 삭제 후보를 생성하세요.")
        return

    st.write(f"추천 결과는 `{cfg.delete_candidates_csv}`에도 저장됩니다.")
    st.dataframe(df[["path", "p_delete", "p_keep", "file_mb"]], use_container_width=True, hide_index=True)

    st.divider()
    st.write("상위 후보 빠른 검수")

    for i, row in df.head(10).iterrows():
        path = row["path"]
        p_delete = float(row["p_delete"])

        with st.container(border=True):
            st.caption(f"p_delete={p_delete:.3f} | {path}")

            try:
                st.image(path, use_container_width=True)
            except Exception:
                st.warning("이미지를 표시할 수 없습니다.")
                continue

            col1, col2, col3 = st.columns(3)

            if col1.button(f"AI 추천대로 버림 0 #{i}", key=f"discard_{i}"):
                upsert_label(
                    cfg,
                    path,
                    label=0,
                    source="ai_recommend_confirmed",
                    reason="ai_delete_candidate",
                    weight=1.5,
                )
                add_xp(cfg, 2)
                st.success("버림 라벨로 저장했습니다.")
                st.rerun()

            if col2.button(f"아님, 남김 1 #{i}", key=f"keep_{i}"):
                upsert_label(
                    cfg,
                    path,
                    label=1,
                    source="ai_recommend_rejected",
                    reason="ai_false_positive",
                    weight=2.0,
                )
                add_xp(cfg, 3)
                st.success("남김 라벨로 저장했습니다. 이 데이터는 AI에게 특히 중요합니다.")
                st.rerun()

            if col3.button(f"스킵 #{i}", key=f"skip_{i}"):
                st.info("스킵했습니다.")
