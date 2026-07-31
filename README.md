# image_quality_classifier

딥러닝을 활용하여 촬영된 이미지의 품질(정상/불량)을 판별하는 이진 분류기(Binary Classifier) 프로젝트입니다.

## 📂 파일 구성 및 쓰임새 (File Description)

### 1. `MobileNetV3 Small.ipynb`[cite: 1]
* **핵심 AI 모델 학습 및 성능 평가 (Model Training & Evaluation)**[cite: 1]
* 실제 이미지 품질을 판별하는 딥러닝 이진 분류 모델을 구축하고 학습시키는 핵심 노트북 파일입니다[cite: 1].
* 경량화 모델인 `MobileNetV3 Small`을 불러와 프로젝트에 맞게 전이 학습(Fine-tuning)을 진행합니다[cite: 1].
* 학습용(Train), 검증용(Validation), 테스트용(Test) 데이터를 나누어 학습을 진행하며, 완료 후 학습 곡선(Accuracy/Loss 그래프)과 오차 행렬(Confusion Matrix)을 통해 모델의 최종 성능을 평가합니다[cite: 1].

### 2. `fail_image_split.ipynb`[cite: 2]
* **학습 전 데이터 전처리 및 자동 분류 (Data Preprocessing & Splitting)**[cite: 2]
* 모델 학습에 앞서, 대량의 원본 데이터셋을 학습하기 좋은 구조(클래스별 폴더)로 정리하는 전처리 스크립트가 포함된 파일입니다[cite: 2].
* JSON 형식의 라벨링 데이터에서 불량(`fail`) 케이스를 찾아내고, 원본 이미지들을 자동으로 `pass`와 `fail` 폴더로 분리하여 복사해 줍니다[cite: 2].
* 이 과정을 통해 머신러닝 학습용 폴더 구조를 손쉽게 세팅할 수 있습니다[cite: 2].
* (참고: 예측된 결과를 바탕으로 모델이 어떤 부분을 보고 판단했는지 확인하는 Grad-CAM 히트맵 시각화 코드도 함께 포함되어 있습니다[cite: 2].)

---

## 📌 주요 특징 (Key Features)

* **경량화 모델 및 전이 학습 (Transfer Learning) 적용**
  * `MobileNetV3Small`을 베이스 모델로 사용하여, 연산량이 적으면서도 높은 정확도를 확보했습니다[cite: 1]. 
  * ImageNet 가중치를 불러온 뒤, 모델의 마지막 25개 층(layer)만 학습 가능하도록 설정하여(Fine-tuning) 맞춤형 데이터셋에 최적화했습니다[cite: 1].

* **고효율 데이터 파이프라인 구축**
  * `tf.keras.utils.image_dataset_from_directory`를 사용하여 대규모 이미지 데이터를 클래스별로 간편하게 로드하고 Train/Validation/Test 셋으로 분할했습니다[cite: 1]. 
  * `cache()`와 `prefetch(AUTOTUNE)`를 적용하여 디스크 I/O 병목 현상을 방지하고 학습 속도를 크게 개선했습니다[cite: 1].

* **직관적인 모델 평가 및 시각화**
  * 학습 과정에서의 정확도(Accuracy) 및 손실(Loss) 변화를 `matplotlib`을 통해 한눈에 확인할 수 있는 그래프로 시각화합니다[cite: 1]. 
  * `scikit-learn`의 `classification_report`와 `confusion_matrix`를 사용하여 클래스별 Precision, Recall, F1-score를 상세하게 분석합니다[cite: 1]. 
  * 모델의 예측 임계값(Threshold)을 0.3으로 조정하여 불량(fail) 탐지율(Recall)을 향상시키는 커스텀 평가 과정을 포함합니다[cite: 1].

* **설명 가능한 AI (Explainable AI, Grad-CAM)**
  * 단순히 정상/불량을 판별하는 것을 넘어, 모델이 이미지의 어느 부분을 보고 불량으로 판단했는지를 시각적으로 보여주는 커스텀 `Grad-CAM` 기능이 구현되어 있습니다[cite: 2]. 
  * 마지막 합성곱 층(Convolutional layer)의 기울기(gradient)를 계산하고, OpenCV를 활용해 원본 이미지 위에 히트맵(Jet colormap)을 합성하여 결과를 출력합니다[cite: 2].

* **데이터 관리 및 전처리 유틸리티 포함**
  * **이상치 탐지:** `find_anomalous_json_files` 함수를 통해 메타데이터(.json)에 기록된 `failure_case_count` 값을 검사하여 손상되거나 비정상적인 데이터를 자동으로 걸러냅니다[cite: 2]. 
  * **자동 폴더 분류:** `separate_images` 스크립트를 통해 원본 이미지와 JSON 파일을 대조하고, 전체 데이터를 'pass'와 'fail' 폴더로 신속하게 자동 분류하여 학습 데이터 세팅을 돕습니다[cite: 2].

---

## 🛠️ 요구 사항 (Prerequisites)
이 코드를 실행하기 위해 필요한 주요 라이브러리는 다음과 같습니다.
* TensorFlow (>= 2.x)[cite: 1, 2]
* OpenCV (`opencv-python`)[cite: 2]
* Matplotlib[cite: 1, 2]
* Scikit-learn[cite: 1]
* NumPy[cite: 1, 2]
