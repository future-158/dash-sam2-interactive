# sam2 추가 


![Demo](assets/sam2.gif)



## 새로운 점

### 현황

sam2을 세그먼테이션에 활용할 때, interactive segmentation으로 사용함. 물론, 라벨링을 한 꺼번에 한 후 추론하는 방식으로도 쓸 수 있지만, 나중에 미흡한 부분에 대해 positive points, negative points를 추가하기가 쉽지 않음
 
### 개선

positive points, negative points가 박스 안에 속하게 구성함 

임의의 box는 0개 이상의 positive points를 포함함
임의의 box는 0개 이상의 negative points를 포함함
모든 positive point, negative point는 정확히 1개의 box에 속해있음

추가로 주시(focusing)을 도입함.
select 모드에서 박스를 클릭하면 그 박스가 주시 상태로 변하고(ui에서는 빨간색 테두리), 오직 주시 상태 박스에 포함된 positive point, negative point만 보여짐

### 기대하는 점

sam2 혹은 이후에 나올 sam3를 라벨링에 도입할 때, interactive segmentation으로만 처리하면, 
- 작업자가 많을 때 실시간 gpu 요청이 많고
- 작업자에게는 딜레이가 신경쓰임

한꺼번에 작업하고 배치 추론을 한 후, 나중에 각각 박스에서 postive points, negative points를 보완하는 식으로 작업

### 설치
```bash
pip install -U -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git

mkdir -p checkpoints
test -f checkpoints/sam2_hiera_tiny.pt || wget --directory-prefix checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt 
```

### 실행
```bash
python tutorials/point-with-box.py 
```
- q: select mode. 박스 주시 변경에 사용
- w: positive mode. 마우스 왼쪽 클릭시 현재 주시 박스 안에 positive point를 생성함
- e: negative mode. 마우스 왼쪽 클릭시 현재 주시 박스 안에 negative point를 생성함
- inference button: 클릭시 추론


### 미비한 점(이후 수정할 점) 
- UI 구성
- 마스크 시각화
- 마스크 내보내는 부분
- fiftyone 제거 


##아래는 이전 내용으로 무관함

[README ENGLISH](README_en.md)

# introduction
![Demo](assets/easy.gif)

링크에서 box, point를 간단히 plotly dash로 구현한 것
[sam2 notebook link](https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb)


```python
# Combining points and boxes
Points and boxes may be combined, just by including both types of prompts to the predictor. Here this can be used to select just the trucks's tire, instead of the entire wheel.

input_box = np.array([425, 600, 700, 875])
input_point = np.array([[575, 750]])
input_label = np.array([0])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)
```
radio button에서 postive point를 선택한 상태로 이미지를 클릭하면 positive point가 찍히는 방식 


## advanced

매번 radio button을 누르기 귀찮기 때문에 bbox로 통일함

사람은 일반적으로 박스를 왼쪽 위에서 오른쪽 아래로 그림

나머지 3가지 방식은
- 오른쪽 위에서 왼쪽 아래로
- 왼쪽 아래에서 오른쪽 위
- 오른쪽 아래에서 왼쪽 위

여기서

- 왼쪽 아래 오른쪽 위로 그릴 경우, 시작점(왼쪽 아래)에 positive point를 생성하게 변경
- 오른쪽 아래 왼쪽 위로 그릴 경우, 시작점(오른쪽 아래) negative point를 생성하게 변경


![Adanvanced Demo](assets/advanced.gif)


## 설치


justfile 사용시
```bash
just install
```

그냥 설치시,
```bash
conda env list | grep  $PWD/venv || conda create -y --prefix $PWD/venv python=3.11 pip ipykernel
conda activate $PWD/venv

pip install -U -r requirements.txt
test -d segment-anything-2 || git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e ".[demo]"

mkdir -p checkpoints
test -f checkpoints/sam2_hiera_tiny.pt || wget --directory-prefix checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt 
```

##  실행
```bash
conda activate venv/
python bbox-app.py
```

## todo:
- multi object 가능하게 변경
