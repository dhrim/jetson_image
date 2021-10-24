
# 물체 탐지 데이터 새로운 데이터 생성

https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md 를 기반으로 함.

<br>

## 프로그램 위치

aarch64/bin/ 혹은  tools/ 혹은 /usr/local/bin/ 에 camera-capture가 있다.

<br>


## 실행 준비

실행하기 전에 다음을 준비한다.

```bash
$ cd /jetson-inference/python/training/detection/ssd/data
$ mkdir new_data
$ echo "class_A
class_B" > new_data/labels.txt
```

classification 처럼 

detection  밑에 data 폴더가 아니라 

detection/ssd 밑에 data 폴더이다.

<br>

## 실행

```bash
$ camera-capture

# 또는
$ camera-capture csi://0
```

![Untitled](images/image3.png)

<br>

## 옵션 설정

다음 옵션을 설정한다.

- Dataset Type : 'Detection'
- Dataset Path : 'jetson-inference/python/training/detection/ssd/data/new_data'
- Class Labels : 'jetson-inference/python/training/detection/ssd/data/new_data/labels.txt'

<br>

## 캡쳐

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