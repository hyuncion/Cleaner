# Cleaner on Google Colab

Colab에서는 Streamlit 앱을 일반 웹앱처럼 바로 호스팅하기보다, Colab proxy로 앱을 여는 방식이 가장 간단합니다.

## 빠른 실행 순서

1. Colab 런타임을 GPU로 바꿉니다. CPU도 가능하지만 embedding 생성이 느립니다.
2. `Cleaner_colab.ipynb` 또는 `notebooks/Cleaner_colab.ipynb`를 Colab에서 엽니다.
3. 이 프로젝트 ZIP을 업로드합니다. GitHub에 올린 뒤에는 notebook의 업로드 셀 대신 clone 방식으로 바꿔도 됩니다.
4. 노트북 셀이 다음을 수행합니다.

```text
의존성 설치
→ demo photos 생성
→ pytest 실행
→ 선택 사항: headless pipeline check
→ Streamlit 서버 실행
→ Colab proxy 창 열기
```

## 앱 안에서 할 일

1. 사진 폴더가 `/content/Cleaner/demo_photos`인지 확인합니다.
2. 왼쪽 사이드바에서 `임베딩 생성/업데이트`를 누릅니다.
3. `스와이프 정리`에서 20장 이상을 `남김=1`, `버림=0`으로 나눕니다.
4. keep/discard가 모두 생긴 뒤 `내 AI 학습시키기`를 누릅니다.
5. `AI 추천 검수` 탭에서 삭제 후보를 확인합니다.

## 권장 테스트 규모

```text
처음: demo photos 100~200장
그다음: 실제 사진 일부 100~500장
성능 확인: 1,000~5,000장
```

사진에는 얼굴, 위치, 문서, 영수증 등 민감 정보가 있을 수 있으므로 Colab에는 샘플만 올리는 것을 권장합니다.
