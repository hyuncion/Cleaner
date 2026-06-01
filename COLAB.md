# Cleaner Colab 사용법

이 프로젝트는 지금 앱을 띄우지 않고 Colab GPU에서 개인화 사진 정리 모델의 최소 루프를 검증하는 형태입니다.

## VS Code에서 실행하는 경우

`Cleaner_colab.ipynb`를 VS Code에서 열어 실행하면 Colab GPU가 아니라 로컬 PC의 Jupyter 커널이 사용됩니다. 따라서 로컬 PC에 NVIDIA CUDA GPU가 없으면 `torch.cuda.is_available()`는 `False`가 맞습니다.

Colab GPU를 쓰려면 같은 `.ipynb` 파일을 Google Colab 웹사이트에서 열고 실행해야 합니다. VS Code는 편집기로 쓰고, 실행은 Colab 브라우저에서 하는 흐름이 가장 안정적입니다.

## 1. 런타임 설정

Colab 메뉴에서 다음을 선택합니다.

```text
런타임 -> 런타임 유형 변경 -> 하드웨어 가속기 GPU
```

그 다음 `Cleaner_colab.ipynb`를 열고 위에서부터 실행합니다.

## 2. 데모 데이터로 확인

노트북의 데모 셀은 다음을 수행합니다.

```text
demo_photos 생성
파일명 기반 demo labels 생성
OpenCLIP embedding 생성
개인화 Logistic Regression 학습
delete_candidates.csv 생성
```

이 단계는 코드가 GPU를 제대로 쓰는지 확인하기 위한 연습입니다.

## 3. 실제 사진 사용

실제 사진은 `/content/my_photos` 같은 폴더에 일부만 업로드해서 시작하는 것을 권장합니다. 처음부터 전체 사진첩을 올리지 마세요.

권장 샘플 크기:

```text
첫 실험: 100~300장
성능 확인: 1,000~5,000장
라벨 시작점: keep/discard 합쳐서 최소 20장, 가능하면 100장 이상
```

## 4. 라벨링 방법

노트북에서는 두 가지 방법을 제공합니다.

1. `start_labeler()` 버튼 위젯 사용
2. CSV 템플릿을 만들고 `label` 열을 직접 채운 뒤 import

라벨 값:

```text
1  = keep
0  = discard
-1 = later
```

`important`는 keep으로 저장하되 weight를 2.0으로 둡니다.

## 5. 출력 파일

기본 출력 폴더는 `/content/cleaner_data`입니다.

```text
labels.csv
embeddings.joblib
personal_keep_delete_model.joblib
delete_candidates.csv
stats.json
events.jsonl
```

`delete_candidates.csv`가 최종 추천 결과입니다. `p_delete`가 높을수록 사용자가 버릴 가능성이 높다고 예측한 사진입니다.

## 6. 주의

사진에는 위치, 얼굴, 문서, 영수증 같은 민감 정보가 들어갈 수 있습니다. Colab 실험에는 반드시 샘플만 올리고, 원본 전체 사진첩을 올리는 방식은 피하세요.
