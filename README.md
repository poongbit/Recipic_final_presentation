# Recipic — 식재료 인식 기반 레시피 추천 앱

<p align="center">
  <img src="https://github.com/user-attachments/assets/1d3c16a6-1fb7-4fde-bd7b-66b8c37ca4c2" width="650" alt="Recipic Project Banner">
</p>

카메라로 식재료를 비추면 기기 안에서 바로 인식하고, 인식된 재료로 만들 수 있는 레시피를 추천하는 Flutter 앱입니다.
2024년 홍익대학교 컴퓨터공학과 졸업 프로젝트(2인 팀, 2023.05–2024.06)로 개발했습니다.

- 시연 영상: https://youtu.be/OP4nHafBLCU
- 담당: 팀장, AI 모델·백엔드

> **영상과 다른 점**
> 발표 영상에서는 인식 모델을 "YOLOv5"로 소개했습니다. 실제로 앱에 탑재된 것은 EfficientNet-Lite4 분류 모델이며, YOLOv5는 학습·변환까지 진행한 뒤 적용하지 않은 실험입니다. 자세한 내용은 아래 "모델"에 정리했습니다.

## 동작 방식

```
카메라 프레임 ─▶ ML Kit 객체 탐지 (물체 위치)
              ─▶ 커스텀 TFLite 분류기 (식재료 종류)
              ─▶ 인식된 재료 목록 ─▶ 앱에 내장된 레시피와 재료 매칭 ─▶ 추천
```

- 추론은 전부 기기 안에서 이루어집니다. 서버로 사진을 보내지 않습니다.
- ML Kit이 물체의 위치를 찾고, 직접 학습한 분류 모델이 그 물체가 어떤 식재료인지 판단합니다.
  (`google_mlkit_object_detection`의 `LocalObjectDetectorOptions`, stream 모드, `classifyObjects: true`)
- 인식 결과는 사용자가 직접 추가·수정할 수 있고, 보유 재료와 겹치는 재료 수를 기준으로 레시피를 보여줍니다.

초기 설계는 사진을 서버로 보내 추론하는 구조였습니다. 통신 지연, 서버 비용, 사용자 사진의 외부 전송을 줄이기 위해 온디바이스 추론으로 바꿨습니다.

## 모델

두 가지 방식을 학습했고, 앱에는 분류 모델 방식을 적용했습니다.

| | 탐지 모델 (실험) | 분류 모델 (앱 적용 방식) |
|---|---|---|
| 구조 | YOLOv5l | EfficientNet-Lite4 (MediaPipe Model Maker) |
| 클래스 | 11종 | 10종 |
| 데이터 | 학습 632장 / 검증 38장 | 843장 → 학습 674 / 검증 85 / 테스트 84 |
| 결과 | mAP50 0.681, mAP50-95 0.385 | 테스트 정확도 92.9% (78/84), 검증 정확도 95.3% |
| TFLite 크기 | 88.2MB (FP16) | 46.8MB (FP32) |
| 앱 적용 | 미적용 | 적용 |

두 모델의 학습 노트북과 실행 로그는 [`notebooks/`](notebooks/)에 있습니다.

- **탐지 모델**: YOLOv5l을 50 epoch 학습하고 TFLite로 변환했습니다. 모델 크기와 검증 정확도를 고려해 앱에는 적용하지 않았습니다.
- **분류 모델**: ImageNet 사전학습 EfficientNet-Lite4의 백본은 고정하고 분류층만 50 epoch 학습했습니다. 입력 300×300, 10종(사과, 바나나, 당근, 옥수수, 오이, 마늘, 레몬, 양파, 감자, 토마토).
- 분류 모델은 클래스 구성을 바꿔가며 여러 번 학습했습니다. 졸업 발표 자료에는 15종 구성(1,253장, 테스트 정확도 92%)이 실려 있고, 저장소의 노트북은 10종 구성의 실행 기록입니다.
- 분류 데이터는 Kaggle 공개 데이터셋(Fruits and Vegetables Image Recognition)의 이미지를 사용했습니다.

### 2026.11 재평가와 경량화

졸업 이후 분류 모델의 평가를 다시 설계하고 모델을 줄였습니다. 자세한 과정과 스크립트는 [`benchmark/`](benchmark/)에 있습니다.

| 모델 | 크기 | 독립 테스트 정확도 (생재료 419장) |
|---|---|---|
| 기존: EfficientNet-Lite4 (FP32) | 46.8MB | 92.6% |
| 신규: EfficientNet-Lite0 미세조정 (INT8) | 4.1MB | 93.6% |

- 기존 정확도 92.9%는 학습 데이터와 같은 출처에서 무작위로 나눈 84장 기준이었고, 학습 데이터 안에 같은 파일이 56장 있었습니다.
- Open Images에서 독립 테스트셋 777장을 만들고 전수 검수해 불량 144장을 걸러냈습니다. 위 표는 그중 통째의 생재료 419장 기준이며, 썰거나 익힌 재료를 포함한 결과는 [`benchmark/`](benchmark/README.md)에 있습니다.
- 실제 사진 4,167장을 추가해 EfficientNet-Lite0를 미세조정하고 INT8로 양자화했습니다. 생재료 기준 정확도를 유지하면서 크기를 46.8MB에서 4.1MB로 줄였습니다.
- 새 모델은 아직 앱에 연동하지 않았습니다.

### 한계

- 실기기(스마트폰)에서의 지연 시간과 메모리 사용량은 측정하지 않았습니다.
- 독립 테스트셋과 추가 학습 데이터가 같은 출처(Open Images)입니다. 실제 사용 환경의 성능은 직접 촬영한 사진으로 다시 재야 합니다.
- 테스트셋 검수는 한 번만 진행했고, 학습 데이터의 라벨 오류는 아직 정제하지 않았습니다.

## 레시피 데이터

만개의레시피에서 수집한 레시피(이름, 재료, 소요 시간, 이미지, 원문 링크)를 사용합니다.
발표 시점에는 Cloud Firestore에 저장해 불러왔고, 현재 저장소에는 27개 레시피가 앱에 내장되어 있습니다(`lib/data/recipes.dart`).

## 프로젝트 구조

```
lib/
├── main.dart
├── ml_object_detection/     # 카메라 스트림, ML Kit 탐지 옵션, 결과 박스 그리기
├── screen/                  # 홈, 인식 재료 확인, 재료 직접 추가, 레시피 목록
├── data/                    # 내장 레시피, 초기 재료 목록
└── const/                   # 공통 UI
notebooks/                   # YOLOv5l, EfficientNet-Lite4 학습 노트북
benchmark/                   # 2026.11 재평가·경량화 스크립트, 테스트셋 목록, 결과
```

## 실행 방법

1. Flutter SDK 3.3.4 이상을 준비합니다.
2. 분류 모델 파일을 `assets/ml/model_test.tflite` 경로에 둡니다. 모델 파일은 용량 문제로 저장소에 포함되어 있지 않습니다.
3. 실행합니다.
   ```bash
   flutter pub get
   flutter run
   ```

카메라를 사용하므로 실제 기기에서 실행해야 합니다.

## 기술 스택

Flutter · Google ML Kit Object Detection · TensorFlow Lite · MediaPipe Model Maker · YOLOv5 (PyTorch) · Firebase (Cloud Firestore)
