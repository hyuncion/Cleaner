# Cleaner

Cleaner는 사진을 정리할수록 사용자의 취향을 학습하는 개인화 AI 사진 정리 실험 프로젝트입니다.

이번 버전은 앱 연동 없이 Colab GPU에서 최소 기능을 검증하기 위한 Python 파이프라인만 남겼습니다. 흐름은 다음과 같습니다.

```text
사진 폴더
-> keep/discard 라벨 CSV
-> OpenCLIP 이미지 embedding 생성
-> Logistic Regression 개인화 모델 학습
-> 삭제 후보 추천 CSV 생성
```

## 현재 구조

```text
Cleaner/
├── Cleaner_colab.ipynb
├── COLAB.md
├── README.md
├── requirements.txt
├── requirements-colab.txt
├── scripts/
│   ├── create_demo_photos.py
│   ├── create_demo_labels.py
│   ├── export_label_template.py
│   ├── import_labels.py
│   └── run_pipeline.py
├── src/
│   └── cleaner/
│       ├── config.py
│       ├── encoder.py
│       ├── embeddings.py
│       ├── features.py
│       ├── labeling.py
│       ├── model.py
│       ├── notebook.py
│       ├── recommendations.py
│       ├── rewards.py
│       ├── storage.py
│       └── utils.py
└── tests/
```

삭제한 것:

- Streamlit 앱 진입점 `app.py`
- `src/cleaner/ui/` 앱 UI 모듈
- Colab에서 Streamlit을 띄우던 예전 노트북 흐름

## 설치

```bash
pip install -e .
```

개발/테스트용:

```bash
pip install -e ".[dev]"
pytest -q
```

## 빠른 로컬 데모

실제 사진 없이 파이프라인만 확인하려면:

```bash
python scripts/create_demo_photos.py --out demo_photos --count 120
python scripts/create_demo_labels.py --photo-dir demo_photos --max 80
python scripts/run_pipeline.py --photo-dir demo_photos --max-embeddings 120 --top-n 20
```

결과물은 기본적으로 `cleaner_data/`에 저장됩니다.

```text
cleaner_data/
├── labels.csv
├── embeddings.joblib
├── personal_keep_delete_model.joblib
├── stats.json
├── events.jsonl
└── delete_candidates.csv
```

## 실제 사진으로 실험하기

앱을 만들기 전 단계에서는 라벨을 CSV로 만들거나 Colab 노트북 위젯으로 붙이면 됩니다.

CSV 템플릿 생성:

```bash
python scripts/export_label_template.py --photo-dir /path/to/photos --out cleaner_data/label_template.csv --max 200
```

`label` 열에 값을 입력합니다.

```text
1  = keep
0  = discard
-1 = later, 학습 제외
```

CSV를 가져온 뒤 학습/추천을 실행합니다.

```bash
python scripts/import_labels.py cleaner_data/label_template.csv
python scripts/run_pipeline.py --photo-dir /path/to/photos --top-n 50
```

Colab/Jupyter에서는 `cleaner.notebook.start_labeler()`로 간단한 버튼형 라벨러를 사용할 수 있습니다.

```python
from cleaner.config import get_config
from cleaner.notebook import start_labeler

cfg = get_config()
start_labeler("my_photos", cfg, limit=100)
```

## 라벨 의미

Cleaner의 제품 아이디어와 맞게 액션은 다음처럼 저장됩니다.

```text
keep       -> label 1
discard    -> label 0
important  -> label 1, weight 2.0
later      -> label -1, 학습 제외
```

모델은 `label`이 0 또는 1인 데이터만 학습합니다.

## Colab GPU

권장 실행은 `Cleaner_colab.ipynb`입니다. 런타임을 GPU로 바꾼 뒤 첫 셀부터 실행하면 됩니다.

핵심 설정은 환경변수로 바꿀 수 있습니다.

```bash
CLEANER_DATA_DIR=/content/cleaner_data
CLEANER_DEVICE=auto
CLEANER_MODEL_NAME=ViT-B-32
CLEANER_PRETRAINED=laion2b_s34b_b79k
CLEANER_BATCH_SIZE=16
```

GPU는 CLIP embedding 생성 단계에서 가장 크게 도움이 됩니다. Logistic Regression 학습은 CPU로도 충분히 빠릅니다.

## MVP 다음 단계

- MobileCLIP 또는 더 작은 encoder 실험
- 중복/비슷한 사진 그룹화
- best-shot 추천
- 추천 검수 결과를 다시 라벨로 반영하는 루프
- 모바일 앱 또는 웹 UI 연결
