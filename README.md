# 스마트홈 IoT 기기 제어를 위한 객체 인식 모델 구현

Jetson Nano에서 딥러닝을 사용한 이미지 처리 교육

<br>

# 프로그램

## 1일차

- [Jetson 셋업](jetson_setup.pdf)
- [Linux 기본 명령어 실습](linux_commands.md)
- Jetson-Inference 프로젝트 설치
    - [소스에서 빌드](jetson_inference/setup_from_source.md)
    - [docker 사용](jetson_inference/setup_by_docker.md)
- 영상 분류 실습
    - [분류(classificaiton)](jetson_inference/execute_classification.md)

<br>

## 2일차
- 물체 탐지 실습
    - [물체 탐지(object detection)](jetson_inference/execute_object_detection.md)

- 영역 분할 실습
    - [영역 분할(segmentation)](jetson_inference/execute_segmentation.md)

- 포즈 추출 실습
    - [포즈 추출(pose estimation)](jetson_inference/execute_pose_estimation.md)

<br>

## 3일차
- 학습 실습
    - [학습 환경 준비](jetson_inference/prepare_training.md)
    - [분류(classification) 학습](jetson_inference/train_classification.md)
    - [물체 탐지(object detection) 학습](jetson_inference/train_object_detection.md)
- [딥러닝의 이해](deep_learning_intro.pptx)

<br>

## 4일차
- AWS 학습 환경 소개, Jupyter 소개
- [AWS 서버에서 분류 학습](jetson_inference/train_classification_on_server.md)
- [AWS 서버에서 물체 탐지 학습](jetson_inference/train_object_detection_on_server.md)
- 커스텀 데이터 AWS 서버에서 분류 학습 : [데이터](data/flowers.zip)

<br>

## 5일차
- 카메라 캡쳐한 데이터 학습
    - [영상 분류](jetson_inference/train_classification_with_custom_data.md)
    - [물체 탐지](jetson_inference/train_object_detection_with_custom_data.md)
- Keras 영상 분류 이해
    - [IRIS 분류](keras/dnn_iris_classification.ipynb)
    - [MNIST 분류](keras/dnn_mnist.ipynb)
- AWS 서버에서 Keras 코드로 분류 학습
    - [노트북](jetson_inference/train_classification_on_server_on_keras.ipynb)
    - [Jetson에 업로드와 분류 실행](jetson_inference/execute_classification_by_uploaded_model.md)

<br>

# 기타

- 흥미로운 딥러닝 결과들 [some_interesting_deep_learning.pptx](some_interesting_deep_learning.pptx)

<br>

# 교육에 필요한 장비
[requirements.md](requirements.md)