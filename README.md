# 딥러닝 (Deep Learning)

딥러닝의 핵심 개념을 단계별로 학습하는 교안 모음입니다.  
이론 설명과 PyTorch 실습 코드를 함께 제공하여, 개념을 이해한 뒤 직접 구현해볼 수 있도록 구성했습니다.

## 환경 설정

- Python 3.12+
- 패키지 관리: [uv](https://docs.astral.sh/uv/)

```bash
# 의존성 설치
uv sync

# Jupyter 실행
uv run jupyter notebook
```

### 주요 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| PyTorch | 딥러닝 프레임워크 |
| NumPy | 수치 연산 |
| Matplotlib | 시각화 |
| scikit-learn | 데이터셋, 전처리 |
| Jupyter | 교안 실행 환경 |

## 교안 목록

| 번호 | 파일 | 주제 | 핵심 내용 |
|------|------|------|----------|
| 01 | [01_DNN_개념.ipynb](01_DNN_개념.ipynb) | DNN (심층 신경망) | 퍼셉트론, 활성화 함수, 순전파, 손실 함수, 역전파, PyTorch 기초 및 DNN 구현 |
| 02 | [02_DNN_훈련.ipynb](02_DNN_훈련.ipynb) | DNN 훈련 기법 | 배치/경사하강법, DataLoader, 과적합 방지(정규화·Dropout·BatchNorm), Early Stopping & LR Scheduler, 옵티마이저 비교, 종합 실습 |
| 03 | [03_RNN_LSTM_시계열.ipynb](03_RNN_LSTM_시계열.ipynb) | RNN · LSTM · Bi-LSTM | 시계열과 슬라이딩 윈도우, RNN 구조와 기울기 소실, LSTM 셀 상태와 3개 게이트, PyTorch LSTM 시계열 예측, 양방향 LSTM, 시퀀스 이진 분류 종합 실습 |

> 교안은 계속 추가될 예정입니다.

## 교안 구성 방식

각 교안은 다음 흐름으로 구성되어 있습니다:

1. **이론 설명** - 개념, 수식, 도표를 통한 핵심 정리
2. **코드 실습** - 개념을 직접 구현하고 결과 확인
3. **시각화** - 학습 과정과 결과를 그래프로 확인
4. **핵심 정리** - 배운 내용 요약 및 다음 주제 예고

## 프로젝트 구조

```
.
├── 01_DNN_개념.ipynb          # 01번 교안 노트북
├── 02_DNN_훈련.ipynb          # 02번 교안 노트북
├── 03_RNN_LSTM_시계열.ipynb   # 03번 교안 노트북
├── images/                    # 교안에서 사용하는 이미지
│   ├── 01_dnn/                #   01번 교안 이미지
│   ├── 02_dnn_training/       #   02번 교안 이미지
│   └── 03_rnn_lstm/           #   03번 교안 이미지
├── pyproject.toml
└── uv.lock
```
