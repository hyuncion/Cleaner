# Cleaner

**Cleaner**는 사진을 스와이프하며 정리할수록 사용자의 취향을 학습하는 개인 AI 사진 정리 MVP입니다.

사용자는 Tinder식으로 사진을 넘기며 `남김=1`, `버림=0` 라벨을 만듭니다. Cleaner는 pretrained CLIP 계열 image encoder로 사진 embedding을 만들고, 사용자의 라벨로 작은 개인화 모델(Logistic Regression)을 학습해 삭제 후보를 추천합니다.

> ⚠️ 이 MVP는 실제 파일을 삭제하지 않습니다. `버림` 라벨과 추천 결과만 저장합니다.

## 핵심 루프

```text
사진 스와이프
→ keep/delete 라벨 생성
→ 개인화 AI 학습
→ 삭제 후보 추천
→ 사용자 검수
→ AI 개선
→ XP/레벨/리포트 보상
```

## 주요 기능

- 사진 폴더 스캔
- pretrained CLIP image embedding 생성
- 스와이프식 `keep/discard` 라벨링
- 사용자별 keep/delete classifier 학습
- AI 삭제 후보 추천 및 검수
- XP, 레벨, 배지, 저장공간 리포트
- Colab 실행용 노트북 포함

## 폴더 구조

```text
Cleaner/
├── app.py
├── Cleaner_colab.ipynb
├── pyproject.toml
├── requirements.txt
├── requirements-colab.txt
├── README.md
├── Makefile
├── scripts/
│   ├── create_demo_photos.py
│   ├── create_demo_labels.py
│   └── colab_pipeline_check.py
├── src/
│   └── cleaner/
│       ├── config.py
│       ├── encoder.py
│       ├── embeddings.py
│       ├── features.py
│       ├── model.py
│       ├── recommendations.py
│       ├── rewards.py
│       ├── storage.py
│       ├── utils.py
│       └── ui/
│           ├── sidebar.py
│           ├── swipe.py
│           ├── recommendations.py
│           └── rewards.py
└── tests/
    ├── test_features.py
    └── test_rewards.py
```

## 로컬 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

개발/테스트 의존성까지 설치하려면:

```bash
pip install -e ".[dev]"
pytest -q
```

## 실행

```bash
streamlit run app.py
```

사진 폴더를 지정한 뒤 다음 순서로 사용합니다.

1. 사이드바에서 `임베딩 생성/업데이트`
2. `스와이프 정리`에서 사진 라벨링
3. 20장 이상, keep/discard가 모두 생기면 `내 AI 학습시키기`
4. `AI 추천 검수`에서 삭제 후보 확인
5. 사용자가 AI 추천을 승인/거절하면 그 결과가 다시 학습 데이터가 됨

## 데모 이미지 만들기

실제 사진 없이 UI만 테스트하고 싶다면:

```bash
python scripts/create_demo_photos.py --out demo_photos --count 120
streamlit run app.py
```

앱에서 `demo_photos` 폴더를 지정하세요.

빠른 파이프라인 검증용 라벨도 자동 생성할 수 있습니다.

```bash
python scripts/create_demo_labels.py --photo-dir demo_photos --max 80
python scripts/colab_pipeline_check.py --photo-dir demo_photos --max-embeddings 40
```

## Colab에서 실행

가장 쉬운 방법은 `Cleaner_colab.ipynb`를 Colab에서 여는 것입니다.

권장 순서:

1. Colab에서 `런타임 → 런타임 유형 변경 → T4 GPU` 선택
2. `Cleaner_colab.ipynb` 실행
3. 이 repo zip 파일을 업로드
4. 데모 사진 생성
5. Streamlit 앱을 Colab proxy로 열기
6. `임베딩 생성/업데이트 → 스와이프 정리 → 내 AI 학습시키기 → AI 추천 검수` 순서로 테스트

Colab에서는 기본 환경변수를 아래처럼 사용합니다.

```bash
CLEANER_DEFAULT_PHOTO_DIR=/content/Cleaner/demo_photos
CLEANER_DATA_DIR=/content/cleaner_data
CLEANER_DEVICE=auto
CLEANER_BATCH_SIZE=16
```

실제 사진을 테스트할 때는 Google Drive에 일부 샘플만 복사해서 사용하세요. 처음부터 전체 사진첩을 올리는 것은 개인정보와 처리 시간 측면에서 권장하지 않습니다.

## 데이터 저장 위치

기본적으로 `cleaner_data/` 폴더에 저장됩니다.

```text
cleaner_data/
├── labels.csv
├── embeddings.joblib
├── personal_keep_delete_model.joblib
├── stats.json
├── events.jsonl
└── delete_candidates.csv
```

## 환경변수

```bash
CLEANER_DEFAULT_PHOTO_DIR=./demo_photos
CLEANER_DATA_DIR=./cleaner_data
CLEANER_DEVICE=auto        # auto, cpu, cuda, mps
CLEANER_MODEL_NAME=ViT-B-32
CLEANER_PRETRAINED=laion2b_s34b_b79k
CLEANER_BATCH_SIZE=16
```

## GPU 필요 여부

GPU는 필수가 아닙니다. CPU만으로도 작동합니다.

다만 시간이 많이 걸리는 부분은 pretrained image encoder로 사진 embedding을 만드는 단계입니다. 이 단계는 GPU가 있으면 훨씬 빠릅니다. 개인화 Logistic Regression 학습은 CPU로 충분합니다.

| 사용량 | 권장 환경 |
|---|---|
| 수백 장 테스트 | CPU 가능 |
| 수천~1만 장 초기 embedding | NVIDIA 6~8GB VRAM 이상이면 쾌적 |
| 수만 장 이상 대량 처리 | 8~12GB VRAM 이상 권장 |
| 모바일 제품화 | MobileCLIP/CoreML/TFLite/ONNX Runtime Mobile 검토 |

## 제품화 시 개선 방향

- MobileCLIP 또는 더 작은 encoder로 교체
- 온디바이스 inference/training
- 비슷한 사진 그룹화: pHash + embedding + timestamp
- best-shot ranking model
- 실제 삭제 대신 앱 내부 휴지통/복원 플로우
- 개인정보 보호: 원본 사진은 서버 전송 금지, opt-in 데이터 기부만 허용
