# Recipic - Object Detection & Recipe Recommendation

<p align="center">
  <img src="https://github.com/user-attachments/assets/1d3c16a6-1fb7-4fde-bd7b-66b8c37ca4c2" width="650" alt="Recipic Project Banner">
</p>

## 1. 개요 (Overview)

**프로젝트명:** 재료 인식을 통한 레시피 추천 어플 (Ingredient Recognition for Recipe Recommendation App)

**목적:**  
2024년 홍익대학교 컴퓨터공학과 졸업 프로젝트로, 사용자가 보유한 재료를 인식하여 적절한 요리 레시피를 추천하는 어플리케이션을 개발했습니다. 이를 통해 사용자는 냉장고나 주방에 있는 재료를 효율적으로 활용하고, 쉽게 레시피를 검색할 수 있습니다.

### 주요 기능:
- **재료 인식:** YOLOv5 모델을 이용하여 사용자가 사진으로 입력한 재료를 인식
- **레시피 추천:** 인식된 재료를 기반으로 관련 레시피를 플러터 앱에서 추천
- **사용자 인터페이스:** 사용하기 쉬운 Flutter 기반의 모바일 어플리케이션 제공

## 2. 제작 과정 (Development Process)

### 2.1 요구사항 분석
- **사용자 요구사항:** 사용자들이 쉽게 재료를 인식시키고, 그 재료로 만들 수 있는 다양한 레시피를 추천받는 기능 구현
- **시스템 요구사항:** 강력한 이미지 인식 모델(YOLOv5)과 사용자 친화적인 인터페이스(Flutter 앱) 활용, 안드로이드 기반 애플리케이션 구현

### 2.2 시스템 설계
- **구조 설계:** YOLOv5 모델을 통해 이미지를 처리하고, Flutter 앱을 통해 결과를 사용자에게 제공하는 구조 설계
- **데이터 흐름:** 
  1. 사용자가 입력한 이미지를 서버로 전송
  2. 서버에서 YOLOv5 모델로 재료 인식
  3. 인식된 재료 정보를 데이터베이스와 비교하여 레시피 추천
  4. Flutter 앱에서 결과 표시

### 2.3 TensorFlow Lite와 Flutter 통합
- **모델 변환:** YOLOv5 모델을 TensorFlow Lite 포맷(.tflite)으로 변환
  ```python
  # YOLOv5 모델을 TFLite로 변환하는 예시 코드
  import torch
  
  # YOLOv5 모델 로드
  model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')
  
  # TorchScript 형식으로 변환
  model.eval()
  traced_model = torch.jit.trace(model, torch.rand(1, 3, 640, 640))
  
  # TFLite 형식으로 내보내기
  model.model = traced_model
  model.export(format='tflite')
  ```

- **Flutter 앱에 통합:**
  1. pubspec.yaml에 필요한 의존성 추가
  ```yaml
  dependencies:
    flutter:
      sdk: flutter
    tflite_flutter: ^0.9.0
    image: ^3.0.2
    camera: ^0.10.0
  ```
  
  2. Android 설정 (android/app/build.gradle)
  ```gradle
  android {
      defaultConfig {
          ...
          ndk {
              abiFilters 'armeabi-v7a', 'arm64-v8a'
          }
      }
  }
  ```
  
  3. iOS 설정 (ios/Podfile)
  ```ruby
  platform :ios, '12.0'
  ```
  
  4. 모델 파일을 assets 폴더에 추가 및 pubspec.yaml에 등록
  ```yaml
  assets:
    - assets/models/yolov5_model.tflite
    - assets/labels/ingredients_labels.txt
  ```

- **모델 로드 및 추론:**
  ```dart
  import 'package:tflite_flutter/tflite_flutter.dart';
  
  class ObjectDetector {
    late Interpreter _interpreter;
    
    Future<void> loadModel() async {
      _interpreter = await Interpreter.fromAsset('assets/models/yolov5_model.tflite');
    }
    
    Future<List<Recognition>> runInference(List<int> imageBytes) async {
      // 이미지 전처리 및 모델 추론 로직
      // ...
      return detectedObjects;
    }
  }
  ```

- **온디바이스 추론의 장점:**
  1. 오프라인 동작 가능 - 인터넷 연결 없이도 기능
  2. 응답 시간 단축 - 서버 통신 지연 없음
  3. 개인정보 보호 - 사용자 데이터가 디바이스를 떠나지 않음
  4. 서버 비용 절감 - 백엔드 인프라 요구사항 감소

### 2.4 모델 학습
- **데이터 수집:** 다양한 음식 재료 이미지 데이터셋 수집
- **데이터 전처리:** 이미지 정규화, 크기 조정, 라벨링 등 데이터 전처리 작업 수행
- **모델 학습:** YOLOv5 모델을 사용하여 학습, 전이 학습을 활용하여 모델의 인식 성능 향상
- **모델 평가:** 학습된 모델을 검증 데이터셋으로 평가하여 정확도 및 성능 측정
- **모델 아키텍처:** YOLOv5의 구조 및 설정 파라미터 조정
- **전이 학습:** ImageNet 등 대형 데이터셋으로 사전 학습된 모델을 활용하여 학습 속도 및 성능 향상
- **하이퍼파라미터 튜닝:** 학습률, 배치 크기 등 하이퍼파라미터 최적화

### 2.5 모델 학습 및 검증
- **훈련 과정:** GPU를 활용한 모델 훈련, 훈련 과정에서의 손실값 및 정확도 모니터링
- **검증 및 테스트:** 검증 데이터셋을 통해 모델의 성능 평가, F1-score, Precision, Recall 등의 지표 사용
- **모델 개선:** 검증 결과를 바탕으로 모델 구조 및 하이퍼파라미터 조정

## 3. 구동 화면 (Demo)

<p align="center">
  <a href="https://youtu.be/OP4nHafBLCU">
    <img src="http://img.youtube.com/vi/OP4nHafBLCU/0.jpg" width="600" alt="Recipic Demo Video">
  </a>
</p>
<p align="center">
  👆 이미지를 클릭하면 데모 영상을 볼 수 있습니다 (Click the image to watch the demo video)
</p>
