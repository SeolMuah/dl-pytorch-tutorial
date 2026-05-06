# 딥러닝 수업 자료

딥러닝의 핵심 개념을 단계별로 학습하는 Jupyter Notebook 교안 모음입니다.
이론 설명, 시각 자료, PyTorch 실습 코드를 함께 제공하여 개념을 확인한 뒤 직접 구현하고 실험할 수 있도록 구성했습니다.

## 학습 환경

- Python 3.12 이상
- 패키지 관리: [uv](https://docs.astral.sh/uv/)
- 실행 환경: Jupyter Notebook

```bash
# 의존성 설치
uv sync

# Jupyter Notebook 실행
uv run jupyter notebook
```

## 주요 라이브러리

| 라이브러리 | 용도 |
| --- | --- |
| PyTorch | 딥러닝 모델 구현 및 학습 |
| NumPy | 수치 연산 |
| Pandas | 데이터 처리 |
| Matplotlib | 시각화 |
| scikit-learn | 예제 데이터셋, 데이터 분할, 전처리 |
| Jupyter | 교안 실행 환경 |

## 교안 목록

| 번호 | 파일 | 주제 | 핵심 내용 |
| --- | --- | --- | --- |
| 01 | [01_DNN_개념.ipynb](01_DNN_개념.ipynb) | DNN 개념 | AI/ML/DL 관계, 퍼셉트론, 활성화 함수, 순전파, 손실 함수, 역전파, PyTorch 기초 |
| 02 | [02_DNN_훈련.ipynb](02_DNN_훈련.ipynb) | DNN 훈련 | 배치 학습, DataLoader, 과적합, 정규화, Dropout, BatchNorm, Early Stopping, LR Scheduler, 옵티마이저 비교 |
| 03 | [03_RNN_LSTM_시계열.ipynb](03_RNN_LSTM_시계열.ipynb) | RNN/LSTM 시계열 | 슬라이딩 윈도우, RNN 구조, 기울기 소실, LSTM 게이트, PyTorch LSTM 예측, Bi-LSTM 분류 |
| 04 | [04_Transformer_시계열.ipynb](04_Transformer_시계열.ipynb) | Transformer 시계열 | Attention, Self-Attention, Q/K/V, Positional Encoding, Multi-Head Attention, Transformer Encoder 예측 |
| 05 | [05_CNN_개념.ipynb](05_CNN_개념.ipynb) | CNN 개념 | 합성곱, 커널, 스트라이드, 패딩, 채널, 풀링, 1D CNN 시계열 분류, 2D CNN 이미지 분류 |

## 교안 구성

각 교안은 다음 흐름을 따릅니다.

1. 이론 설명: 개념, 수식, 도표로 핵심 원리를 정리합니다.
2. 코드 실습: PyTorch와 주요 라이브러리로 직접 구현합니다.
3. 시각화: 데이터, 학습 과정, 예측 결과를 그래프로 확인합니다.
4. 종합 실습: 한 챕터의 주요 개념을 연결해 작은 문제를 끝까지 해결합니다.
5. 핵심 정리: 배운 내용과 다음 확장 방향을 정리합니다.

## 추가 실습 파일

| 파일 | 설명 |
| --- | --- |
| [practice_05_cnn_training.py](practice_05_cnn_training.py) | sklearn digits 데이터를 2D CNN으로 분류하는 단독 실행 실습 스크립트 |

실행 예시:

```bash
uv run python practice_05_cnn_training.py
```

## 프로젝트 구조

```text
.
├── 01_DNN_개념.ipynb
├── 02_DNN_훈련.ipynb
├── 03_RNN_LSTM_시계열.ipynb
├── 04_Transformer_시계열.ipynb
├── 05_CNN_개념.ipynb
├── images/
│   ├── 01_dnn/
│   ├── 02_dnn_training/
│   ├── 03_rnn_lstm/
│   ├── 04_transformer_timeseries/
│   └── 05_cnn/
├── practice_05_cnn_training.py
├── pyproject.toml
└── uv.lock
```

## 참고

- 노트북은 순서대로 학습하는 것을 기준으로 작성되어 있습니다.
- 이미지 파일은 각 교안의 설명 도표로 사용되므로 `images/` 디렉터리 구조를 유지해야 합니다.
- 일부 실습은 실행 시 학습 시간이 걸릴 수 있습니다.
