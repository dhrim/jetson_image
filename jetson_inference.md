# jetson-inference 프로젝트

- https://github.com/dusty-nv/jetson-inference
- NVIDIA의 공식 프로젝트. 
- 소스에서 빌드할 수도 있고, 이미 빌드된 docker를 사용할 수도 있다.
- 특정 모델을 선택하여 다운로드하여 실행함.
- 생성된 python 실행파일로 물체 분류(classification), 물체 탐지(object detection), 영역 분할(segmentation), 깊이 탐지(depth detection) 실행

<br>

# 프로젝트 전체 내용

- jetson 셋업. 소스 빌드하거나 docker를 사용
- 분류, 탐지, 분할, 포즈추출 할 수 있고
- 데이터 가져와서 분류 학습할 수 있고
- 혹은 분류, 탐지(SSD) 데이터 새로 만들어 학습

<br>

# Jetson 셋업

[Jetson 셋업](jetson_setup.pdf)의 방법으로 Jetson을 부팅상태로 한다.

<br>

# 소스에서 빌딩

[https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md)를 기반으로 함.

<br>

## 필요 프로그램, 라이브러리 설치

```bash
$ sudo apt-get update
$ sudo apt-get install git cmake libpython3-dev python3-numpy
```

<br>

## 프로젝트 다운로드

```bash
$ git clone https://github.com/dusty-nv/jetson-inference
$ cd jetson-inference
$ git submodule update --init
```

<br>

## cmake 빌드

```bash
$ cd ~/jetson-inference
$ mkdir build
$ cd build
$ cmake ../
```

<br>

## 모델 다운로드

cmake가 종료되면 자동으로 다음이 실행된다.

![Untitled](jetson_inference_images/image1.png)

다운로드 할 모델을 스페이스바 눌러서 선택하고 엔터 클릭하여 다운로드 실행.

수동으로 실행시키려면 다음을 실행.

```bash
$ cd ~/jetson-inference/tools
$ ./download-models.sh
```

<br>

## PyTorch 설치

다운로드 완료되면 다음이 자동으로 실행된다.

![Untitled](jetson_inference_images/image2.png)

아래에 있는 'PyTorch v.1.4.0 for Python 3.6'에서 스페이스바 클릭, 엔터 클릭.

자동으로 실행되지 않으면 다음으로 실행시킨다.

```bash
$ cd ~/jetson-inference/build
$ ./install-pytorch.sh
```

<br>

## 컴파일/설치

```bash
$ make
$ sudo make install
$ sudo ldconfig
```

빌드된 것들이 aarch64/bin에 설치됨.

<br>

## 실행

파일 대상 분류

```bash
$ cd aarch64/bin
$ ./imagenet-console.py images/humans_0.jpg result.jpg
```

결과는 result.jpg로 저장된다. ubuntu의 윈도우 탐색기(?)로 확인 가능하다.
![Untitled](jetson_inference_images/image4.png)

<br>

실시간 카메라 대상 분류

```bash
$ cd aarch64/bin
$ ./imagenet-camera.py 
```

윈도우 창이 뜨면서 실시간으로 분류 실행됨.

<br>

# Docker로 환경 준비

[https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md)를 기반으로 함.

<br>

## 프로젝트 다운로드

```bash
$ git clone --recursive https://github.com/dusty-nv/jetson-inference

```

<br>

## docker 실행

```bash
$ cd jetson-inference
$ docker/run.sh
```

실행되고 나면 모델 다운로드 화면이 뜬다.

PyTorch 설치 화면은 안뜬다.

마운트된 디렉토리 구조

```bash
/jetson-inference/
	data/networks/
	python/
		training/
			classification/
				data/
				models/
			detection/
				data/
				models/
```

<br>

## 동작 확인

```bash
$ cd build/aarch64/bin
$ ./imagenet images/jellyfish.jpg images/test/result.jpg
```

<br>

## build/aarch64/bin 밑의 파일 들

```bash
detectnet-camera.py  # detectnet.py와 동일. 카메라 동영상에 대해
detectnet-console.py # detectnet.py와 동일. 파일에 대해
detectnet.py         # 카메라 동영상 혹은 파일에 대해

imagenet-camera.py  # imagenet.py와 동일. 카메라 동영상에 대해
imagenet-console.py # imagenet.py와 동일. 파일에 대해
imagenet.py         # 카메라 동영상 혹은 파일에 대해

segnet-camera.py    # segnet.py와 동일
segnet-console.py   # segnet.py와 동일
segnet.py           # 카메라 동영상 혹은 파일에 대해

posenet.py          # 포즈 추출

video-viewer.py
caemra-viewr.py

my-detection.py     # object detection 코드 template
my-recognition.py   # classification 코드 template
```

<br>

# 분류(classification)

## default 모델로 실행

default model로 실행

```bash
$ ./imagenet.py images/orange_0.jpg images/test/output_0.jpg
```

<br>

## 특정 모델로 실행

```bash
$ ./imagenet.py --network=resnet-18 images/jellyfish.jpg images/test/output_jellyfish.jpg
```

모델 다운로드는 tools/download-models.sh로 할 수 있다.

각 모델의 이름은 다음과 같다.

[분류 모델 테이블](jetson_inference%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20c7362a65de6c482aa0b7a2584e8432e9/%E1%84%87%E1%85%AE%E1%86%AB%E1%84%85%E1%85%B2%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%90%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%87%E1%85%B3%E1%86%AF%205a0667649d8042d2a39862aae24fbe53.csv)

<br>

## 카메라에 대하여 실행

[https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-camera-2.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-camera-2.md)

```bash
./imagenet.py csi://0
```

docker 실행될 때 V2L2에 /dev/video0이 잡히지만 다음으로 실행하면 안된다.

```bash
./imagenet.py /dev/video0
```

<br>

## 동영상 파일에 대하여 실행

```bash
$ wget https://nvidia.box.com/shared/static/tlswont1jnyu3ix2tbf7utaekpzcx4rc.mkv -O jellyfish.mkv

$ ./imagenet.py --network=resnet-18 jellyfish.mkv images/test/jellyfish_resnet18.mkv
```

<br>

## python 코드 작성

[https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-example-python-2.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-example-python-2.md)

작업 폴더 mount

```bash
# run these commands outside of container
$ cd ~/

$ mkdir my-recognition-python
$ cd my-recognition-python

$ touch my-recognition.py
$ chmod +x my-recognition.py

$ wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/black_bear.jpg 
$ wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/brown_bear.jpg
$ wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/polar_bear.jpg
```

외부의 ~/my_recognition-python 디렉토리를 docker 내의 /my-recognition-python으로 마운트.

```bash
$ cd ~/jetson-inference

$ docker/run.sh --volume ~/my-recognition-python:/my-recognition-python 
```

[my-recognition.py](http://my-recognition.py) 내용

```bash
#!/usr/bin/python3

import jetson.inference
import jetson.utils

import argparse

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect.")
args = parser.parse_args()

# 이미지 파일 로딩
img = jetson.utils.loadImage(args.filename)

# 모델 로딩
net = jetson.inference.imageNet(args.network)

# 분류 실행
class_idx, confidence = net.Classify(img)

# 카테고리 설명 구하기
class_desc = net.GetClassDesc(class_idx)

# 결과 출력
print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))
```

<br>

# 물체탐지(object detection)

## default 모델 사용

```bash
./detectnet.py images/peds_0.jpg images/test/output.jpg
```

<br>

## 특정 모델 사용

```bash
$ ./detectnet.py --network=ssd-mobilenet-v2 images/peds_0.jpg images/test/output.jpg
```

[물체 탐지 모델 테이블](jetson-inference%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20c7362a65de6c482aa0b7a2584e8432e9/%E1%84%86%E1%85%AE%E1%86%AF%E1%84%8E%E1%85%A6%20%E1%84%90%E1%85%A1%E1%86%B7%E1%84%8C%E1%85%B5%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%90%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%87%E1%85%B3%E1%86%AF%20c3491a57d80c46708bbf676759677750.csv)

<br>

## 디렉토리 내 여러개 파일

```bash
$ ./detectnet.py "images/peds_*.jpg" images/test/peds_output_%i.jpg
```

<br>

## 동영상 파일

container 밖에서 파일 준비

```bash
$ cp /usr/share/visionworks/sources/data/pedestrians.mp4 data/images/
$ cp /usr/share/visionworks/sources/data/parking_ssd.avi data/images/
```

```bash
$ ./detectnet.py images/pedestrians.mp4 images/test/pedestrians_ssd.mp4

$ ./detectnet.py images/parking.avi images/test/parking_ssd.avi
```

<br>

## 카메라 사용

```bash
$ ./detectnet.py csi://0
```

<br>

## detectnet.py 내용

```bash
import jetson.inference
import jetson.utils

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.")

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# 파일이든 동영상이든 카메라든 관계 없다.
# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

# process frames until the user exits
while True:
	# capture the next image
	img = input.Capture()

	# 탐지 결과는 img에 그려져 있고, 개별 결과는 detections에 담겨 있다.
	detections = net.Detect(img, overlay=opt.overlay)

	# 콘솔에 축력하고
	print("detected {:d} objects in image".format(len(detections))
	for detection in detections:
		print(detection)

	# 결과를 출력하고
	output.Render(img)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break
```

<br>

## python 코드 작성

```bash
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
```

<br>

# 영역분할(segmentation)

<br>

## default 모델 사용

```bash
$ ./segnet.py images/city_0.jpg images/test/output.jpg

$ ./segnet.py --alpha=200 images/city_0.jpg images/test/output.jpg

$ ./segnet.py --visualize=maskimages/city_0.jpg images/test/output.jpg
```

<br>

## 특정 모델 사용

```bash
$ ./segnet.py --network=fcn-resnet18-cityscapes images/city_0.jpg images/test/output.jpg
```

[영역분할 모델 테이블](jetson-inference%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20c7362a65de6c482aa0b7a2584e8432e9/%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%8B%E1%85%A7%E1%86%A8%E1%84%87%E1%85%AE%E1%86%AB%E1%84%92%E1%85%A1%E1%86%AF%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%90%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%87%E1%85%B3%E1%86%AF%209cb66142ac654d21aa6557977acb5769.csv)

<br>

### 폴더 파일 전체에 대해

```bash
$ ./segnet.py --network=fcn-resnet18-sun "images/room_*.jpg" images/test/room_output_%i.jpg
```

<br>

## 카메라 동영상에 대해

```bash
~ ./senet.py csi://0
```

<br>

# 포즈추출(pose estimation)

<br>

## default 모델로

```bash
./posenset.py images/human_9.jpg images/test/human_9_pose.jpg
```

<br>

## 특정 모델로

```bash
./posenset.py --network=resnet18-body images/human_9.jpg images/test/human_9_pose.jpg
```

[포즈추출 모델 테이블](jetson-inference%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20c7362a65de6c482aa0b7a2584e8432e9/%E1%84%91%E1%85%A9%E1%84%8C%E1%85%B3%E1%84%8E%E1%85%AE%E1%84%8E%E1%85%AE%E1%86%AF%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%90%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%87%E1%85%B3%E1%86%AF%206186f0640b25481bb6e2a8edea7760d3.csv)

<br>

## 폴더 파일 전체에 대해

```bash
$ ./posenet.py "images/humans_*.jpg" images/test/pose_humans_%i.jpg
```

<br>

## 카메라 동영상에 대해

```bash
./posenet.py csi://0
```

<br>

## 코드로 값 구하기

```bash
poses = net.Process(img)

for pose in poses:
    # find the keypoint index from the list of detected keypoints
    # you can find these keypoint names in the model's JSON file, 
    # or with net.GetKeypointName() / net.GetNumKeypoints()
    left_wrist_idx = pose.FindKeypoint('left_wrist')
    left_shoulder_idx = pose.FindKeypoint('left_shoulder')

    # if the keypoint index is < 0, it means it wasn't found in the image
    if left_wrist_idx < 0 or left_shoulder_idx < 0:
        continue
	
    left_wrist = pose.Keypoints[left_wrist_idx]
    left_shoulder = pose.Keypoints[left_shoulder_idx]

    point_x = left_shoulder.x - left_wrist.x
    point_y = left_shoulder.y - left_wrist.y

    print(f"person {pose.ID} is pointing towards ({point_x}, {point_y})")
```

<br>

# 깊이 측정

<br>

## default 모델로

```bash
./depthnet.py images/room_1.jpg images/test/room_1_depth.jpg
```

<br>

## 폴더 파일 전체에 대해

```bash
$ ./depthnet.py "images/room_*.jpg" images/test/room_%i_depth.jpg
```

<br>

## 카메라 동영상에 대해

```bash
./depthnet.py csi://0
```

<br>

## 코드로 값 구하기

```bash
import jetson.inference
import jetson.utils

import numpy as np

# 모델 로딩
net = jetson.inference.depthNet()

# depthNet re-uses the same memory for the depth field,
# so you only need to do this once (not every frame)
depth_field = net.GetDepthField()

# cudaToNumpy() will map the depth field cudaImage to numpy
# this mapping is persistent, so you only need to do it once
# 요 어레이에 depth가 담긴다.
depth_numpy = jetson.utils.cudaToNumpy(depth_field)

print(f"depth field resolution is {depth_field.width}x{depth_field.height}, format={depth_field.format})

while True:
    img = input.Capture()	# assumes you have created an input videoSource stream
    net.Process(img)
    jetson.utils.cudaDeviceSynchronize() # wait for GPU to finish processing, so we can use the results on CPU
	
    # find the min/max values with numpy
    min_depth = np.amin(depth_numpy)
    max_depth = np.amax(depth_numpy)
```

<br>

# 학습 환경 준비

<br>

## 스왑 공간 마운팅

docker 밖에서

```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 4G /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
```

/etc/tstab에 다음줄을 추가

```bash
/mnt/4GB.swap  none  swap  sw 0  0
```

<br>

## PyTorch 준비

docker 안에서 다음을 실행하여 결과 확인.

```bash
$ python

>>> import torch
>>> print(torch.__version__)
>>> print('CUDA available: ' + str(torch.cuda.is_available()))
>>> a = torch.cuda.FloatTensor(2).zero_()
>>> print('Tensor a = ' + str(a))
>>> b = torch.randn(2).cuda()
>>> print('Tensor b = ' + str(b))
>>> c = a + b
>>> print('Tensor c = ' + str(c))
```

만약 설치가 안되어 있다면 다음으로 설치.

```bash
$ cd jetson-inference/build
$ ./install-pytorch.sh
```

<br>

# cat_dog 분류 학습

[https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-cat-dog.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-cat-dog.md)

<br>

## 데이터 다운로드

```bash
$ cd jetson-inference/python/training/classification/data
$ wget https://nvidia.box.com/shared/static/o577zd8yp3lmxf5zhm38svrbrv45am3y.gz -O cat_dog.tar.gz
$ tar xvzf cat_dog.tar.gz
```

파일 구조는 다음과 같다.

```bash
cat_dog/
	labels.txt
	train/
		cat/
		dog/
	val/
		cat/
		dog/
	test/
		cat/
		dog/
```

## 학습 실행

```bash
$ cd jetson-inference/python/training/classification
$ python3 train.py --model-dir=models/cat_dog data/cat_dog
```

epoch 당 7~8분, 35 epoch에 4시간

모델 저장 위치는 

```bash
# docker 밖에서 
~/jetson-inference/python/training/classification/models/cat_dog/

# docker 안에서 
/jetson-inference/python/training/classification/models/cat_dog/
	checkpoint.pth.tar
	model_best.pth.tar
```

<br>

## ONNX 포멧으로 converting

```bash
python3 onnx_export.py --model-dir=models/cat_dog
```

jetson-inference/python/training/classification/models/cat_dog/ 아래에 

resnet18.onnx 파일이 생성된다.

<br>

## 분류 실행

```bash
NET=models/cat_dog
DATASET=data/cat_dog

imagenet.py --model=$NET/cat_dog/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/cat/01.jpg cat.jpg
```

여기서 실행되는 imagenet.py는 /usr/local/bin/imagenet.py이다.

<br>

## 전체 폴더 구조

```bash
# docker 안에서
/usr/local/bin/
	imagenet.py
	detectnet.py
	posenet.py

/jetson-inference/python/training/classification/
	train.py
	onnx_export.py
	data/
		cat_dog.tar.gz
		cat_dog/
			labels.txt
			train/
				cat/
				dog/
			val/
				cat/
				dog/
			test/
				cat/
				dog/
	models/
		cat_dog/
			checkpoint.pth.tar
			model_best.pth.tar
			resnet18.onnx
```

data/cat_dog/labels.txt 내용

```bash
cat
dog
```

<br>

# Plant 분류 학습

[https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-plants.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-plants.md)

<br>

## 데이터 다운로드

```bash
$ cd jetson-inference/python/training/classification/data
$ wget https://nvidia.box.com/shared/static/vbsywpw5iqy7r38j78xs0ctalg7jrg79.gz -O PlantCLEF_Subset.tar.gz
$ tar xvzf PlantCLEF_Subset.tar.gz
```

데이터 폴더 구조

```bash
PlantCLEF_Subset/
	labels.txt
	test/
	train/
		ash/
		beech/
		...
	val/
		ash/
		beech/
		...

```

<br>

## 학습 실행

```bash
$ cd jetson-inference/python/training/classification
$ python3 train.py --model-dir=models/plants data/PlantCLEF_Subset
```

생성된 모델 파일

```bash
# docker 밖에서 
~/jetson-inference/python/training/classification/models/plants/

# docker 안에서 
/jetson-inference/python/training/classification/models/plants/
	checkpoint.pth.tar
	model_best.pth.tar
```

<br>

## ONNX 포멧으로 converting

```bash
python3 onnx_export.py --model-dir=model/plants
```

jetson-inference/python/training/classification/models/plants/ 아래에 

resnet18.onnx 파일이 생성된다.

<br>

## 분류 실행

```bash
NET=models/plants
DATASET=data/PlantCLEF_Subset

imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/cattail.jpg cattail.jpg
```

<br>

## 전체 폴더 구조

```bash
# docker 안에서
/usr/local/bin/
	imagenet.py
	detectnet.py
	posenet.py

/jetson-inference/python/training/classification/
	train.py
	onnx_export.py
	data/
		PlantCLEF_Subset.tar.gz
		PlantCLEF_Subset/
			labels.txt
			train/
				ash/
				beech/
				...
			val/
				ash/
				beech/
				...
			test/
	models/
		platns/
			checkpoint.pth.tar
			model_best.pth.tar
			resnet18.onnx
```

data/PlantCLEF_Subset/labels.txt 내용

```bash
ash
beech
...
```

<br>

# 새로운 데이터 생성

[https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect.md)

<br>

## 프로그램 위치

aarch64/bin/ 혹은  tools/ 혹은 /usr/local/bin/ 에 camera-capture가 있다.

<br>

## 분류 데이터

<br>

### 실행 준비

실행하기 전에 다음을 준비한다.

```bash
$ cd /jetson-inference/python/training/classification/data
$ mkdir new_data
$ echo "class_A
class_B" > new_data/labels.txt
```

<br>

### 실행

```bash
$ camera-capture

# 또는
$ camera-capture csi://0
```

![Untitled](jetson-inference%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20c7362a65de6c482aa0b7a2584e8432e9/Untitled%202.png)

<br>

### 옵션 설정

다음 옵션을 설정한다.

- Dataset Type : 'Classification'
- Dataset Path : 'jetson-inference/python/training/classification/data/new_data'
- Class Labels : 'jetson-inference/python/training/classification/data/new_data/labels.txt'

<br>

### 캡쳐

옵션을 설정하고

- Current Set : 'train'
- Current Class : 'class_A'

버튼 'Capture (space)'를 클릭한다. 클릭하는데로 jpg파일이 생성된다.

<br>

## 물체 탐지 데이터

[https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md)

<br>

### 실행 준비

실행하기 전에 다음을 준비한다.

```bash
# $ cd /jetson-inference/python/training/classification/data
$ cd /jetson-inference/python/training/detection/ssd/data
$ mkdir new_data
$ echo "class_A
class_B" > new_data/labels.txt
```

classification 처럼 

detection  밑에 data 폴더가 아니라 

detection/ssd 밑에 data 폴더이다.

<br>

### 실행

```bash
$ camera-capture

# 또는
$ camera-capture csi://0
```

![Untitled](jetson-inference%20%E1%84%91%E1%85%B3%E1%84%85%E1%85%A9%E1%84%8C%E1%85%A6%E1%86%A8%E1%84%90%E1%85%B3%20c7362a65de6c482aa0b7a2584e8432e9/Untitled%202.png)

<br>

### 옵션 설정

다음 옵션을 설정한다.

- Dataset Type : 'Detection'
- Dataset Path : 'jetson-inference/python/training/detection/ssd/data/new_data'
- Class Labels : 'jetson-inference/python/training/detection/ssd/data/new_data/labels.txt'

<br>

### 캡쳐

옵션을 설정하고

- Current Set : 'train'
- Current Class : 'class_A'

버튼 'Freeze/Edit (space)'를 클릭한다. 

화면상에서 레이블링하고 'Save (S)'를 클릭한다.

<br>

### 버그?

새로 Freeze하면 레이블링이 안된다. 다시 Freeze해야 한다.

<br>

### 생성된 데이터 파일들

다소 복잡하다.

```bash
new_data/
	labels.txt
	Annotations/
		20210903-231110.xml
		20210903-231112.xml
		...
	ImageSets/
		Main/
			train.txt
			trainval.txt
	JPEGImages/
		20210903-231110.jpg
		20210903-231112.jpg
		...
```

Annotasions/20210903-231110.xml 내용

```bash
<annotaion>
	<filename>20210903-231110.jpg</filename>
	<folder>new_data</folder>
	<source>
		<database>new_data></database>
		<annotation>custom</annotation>
		<image>custom</image>
	</source>
	<size>
		<width>1280</width>
		<height>720></height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>class_A</name>
		<pose>unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>290</xmin>
			<ymin>60</ymin>
			<xmax>777</xmax>
			<ymax>356</ymax>
		</bndbox>
	</object>
</annotation>
```

ImageSets/Main/train.txt 내용

```bash
20210903-231110
20210903-231112
...
```

ImageSets/Main/trainval.txt 내용

```bash
20210903-231110
20210903-231112
...
```

학습할 때는 train-ssd.py를 사용한다.