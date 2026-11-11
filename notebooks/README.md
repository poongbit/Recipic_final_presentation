# 학습 노트북

## yolov5l_train_tflite_export.ipynb

2024년 5월 8일 Google Colab(Tesla T4)에서 실행한 YOLOv5 학습·변환 기록입니다. 실행 결과(로그)를 그대로 포함하고 있습니다.

| 항목 | 값 |
| --- | --- |
| 모델 | YOLOv5l (사전학습 가중치 `yolov5l.pt`에서 전이학습) |
| 클래스 | 11종 (bean sprouts, beef, chicken, egg, fork, garlic, green onion, kimchi, onion, potato, spam) |
| 데이터 | 학습 632장 / 검증 38장 (바운딩 박스 라벨) |
| 학습 설정 | 이미지 640, 배치 16, 50 에폭 |
| 검증 결과 | mAP50 0.681, mAP50-95 0.385, Precision 0.636, Recall 0.622 |
| 변환 | `best.pt` (88.7MB) → `best-fp16.tflite` (88.2MB) |

### 참고

- 이 노트북의 모델은 **앱에 탑재된 모델이 아닙니다.** 현재 앱 코드(`lib/ml_object_detection`)는 ML Kit 객체 탐지에 커스텀 이미지 분류 모델(TFLite)을 연결하는 방식을 사용합니다.
- YOLOv5l은 모델 크기(약 88MB)와 검증 정확도를 고려해 앱에는 적용하지 않았습니다.
- 데이터셋은 Google Drive 경로를 참조하며 이 저장소에는 포함되어 있지 않습니다.
- 원본 파일명은 `Tensorflow_lite_label_image.ipynb`입니다.

## efficientnet_lite4_classifier_train.ipynb

2024년 5월 Google Colab(GPU)에서 실행한 이미지 분류 모델 학습·변환 기록입니다. 마지막 실행(5월 30일)의 결과를 그대로 포함하고 있습니다.

| 항목 | 값 |
| --- | --- |
| 도구 | MediaPipe Model Maker 0.2.1.4 (`image_classifier`) |
| 모델 | EfficientNet-Lite4, 백본 고정 후 분류층만 학습 (전체 11,850,746개 중 학습 파라미터 12,810개) |
| 클래스 | 10종 (apple, banana, carrot, corn, cucumber, garlic, lemon, onion, potato, tomato) |
| 데이터 | 843장 (클래스당 68~94장) → 학습 674 / 검증 85 / 테스트 84 |
| 학습 설정 | 50 에폭, 배치 2 (기본값) |
| 검증 결과 | 정확도 95.3% (최고 96.5%, 8 에폭) |
| 테스트 결과 | 정확도 92.9% (78/84) |
| 변환 | `exported_model/model.tflite` (FP32, 약 46.8MB) |

### 참고

- 앱은 이 방식으로 학습한 분류 모델을 ML Kit 객체 탐지의 커스텀 분류기로 사용합니다.
- 같은 노트북으로 클래스 구성을 바꿔가며 여러 번 학습했고, 저장된 출력은 10종 구성의 마지막 실행입니다. 졸업 발표 자료의 15종 구성(1,253장, 테스트 정확도 92%) 실행 기록은 남아 있지 않습니다.
- 검증 정확도는 8 에폭 이후 거의 변하지 않았습니다.
- 데이터는 Kaggle 공개 데이터셋(Fruits and Vegetables Image Recognition)의 이미지이며 이 저장소에는 포함되어 있지 않습니다.
- 원본 파일명은 `2023_Training_Image_Classification_Models.ipynb`입니다.
