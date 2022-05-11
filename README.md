





## 环境配置

本项目的环境和代码在新大陆集团提供的人工智能开发平台进行测试，经测试以下所有脚本均可以正确运行。

本项目在python=3.6.5 Linux（新大陆平台）下测试

我们推荐使用conda配置，否则直接安装rknn-toolkit可能会出现意想不到的错误

conda配置

```python
conda create -n traffic python=3.6.5
conda activate traffic
```

安装依赖

```python
pip install -r requirements.txt
```

安装rknn-toolkit

```python
pip install rknn_toolkit-1.7.1-cp36-cp36m-linux_x86_64.whl
```



## 目录结构

```
│  dataset.txt
│  detect.py
│  onnx_to_rknn.py
│  prepare_data.py
│  README.md
│  requirements.txt
│  result.txt
│  testEnvironment.py
│  traffic_detect_image.py
│  traffic_detect_video.py
│  train.py
│  video.mp4
│
├─.idea
│  │  .gitignore
│  │  .name
│  │  modules.xml
│  │  traffic-detection.iml
│  │  vcs.xml
│  │  workspace.xml
│  │
│  └─inspectionProfiles
│          profiles_settings.xml
│          Project_Default.xml
│
├─data
│  │  coco.yaml
│  │  coco128.yaml
│  │  hyp.finetune.yaml
│  │  hyp.scratch.yaml
│  │  traffic.yaml
│  │  voc.yaml
│  │
│  ├─images
│  │      002066.jpg
│  │      bus.jpg
│  │      zidane.jpg
│  │
│  └─scripts
│          get_coco.sh
│          get_voc.sh
│
├─models
│  │  common.py
│  │  experimental.py
│  │  export.py
│  │  yolo.py
│  │  yolov5l.yaml
│  │  yolov5m.yaml
│  │  yolov5s.yaml
│  │  yolov5s_traffic.yaml
│  │  yolov5x.yaml
│  │  yolov5_rknn_640x640.yaml
│  │  __init__.py
│  │
│  ├─hub
│  │      anchors.yaml
│  │      yolov3-spp.yaml
│  │      yolov3-tiny.yaml
│  │      yolov3.yaml
│  │      yolov5-fpn.yaml
│  │      yolov5-p2.yaml
│  │      yolov5-p6.yaml
│  │      yolov5-p7.yaml
│  │      yolov5-panet.yaml
│  │
│  └─__pycache__
│          common.cpython-36.pyc
│          experimental.cpython-36.pyc
│          yolo.cpython-36.pyc
│          __init__.cpython-36.pyc
│
├─utils
│  │  activations.py
│  │  autoanchor.py
│  │  datasets.py
│  │  general.py
│  │  google_utils.py
│  │  loss.py
│  │  metrics.py
│  │  plots.py
│  │  torch_utils.py
│  │  __init__.py
│  │
│  ├─google_app_engine
│  │      additional_requirements.txt
│  │      app.yaml
│  │      Dockerfile
│  │
│  └─__pycache__
│          activations.cpython-36.pyc
│          autoanchor.cpython-36.pyc
│          datasets.cpython-36.pyc
│          general.cpython-36.pyc
│          google_utils.cpython-36.pyc
│          loss.cpython-36.pyc
│          metrics.cpython-36.pyc
│          plots.cpython-36.pyc
│          torch_utils.cpython-36.pyc
│          __init__.cpython-36.pyc
│
└─weights
        download_weights.sh
        traffic_500.pt
        traffic_500.rknn
        yolov5s.pt
        yolov5s.rknn
```



## 快速开始

### 1、模型训练

在data/traffic.yaml中设置数据集位置

```python
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: /gemini/data-1/VOCdevkit/images/train/  # xxx images
val: /gemini/data-1/VOCdevkit/images/val/  # xxx images
```

使用预训练模型yolov5s.pt进行训练

```python
python train.py --weights/yolov5s.pt
```



### 2、模型转换

**pytorch模型转换为onnx模型**

```python
python models/export.py --weights "xxx.pt"
```

排错

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
如遇这种错误，请运行以下命令：pip install opencv-python-headless

ONNX export failure: Your model ir_version is higher than the checker's.
如遇此类错误，可执行以下命令，将onnx升级到1.7.0 ：pip install onnx==1.7.0
```



**onnx模型转换为rknn模型**

修改onnx模型和rknn模型地址

```python
ONNX_MODEL = 'weights/traffic_500_640x640.onnx'
RKNN_MODEL = 'weights/traffic_500.rknn'
```

执行转换脚本

```python
python onnx_to_rknn.py 
```

在weights目录下获得traffic_500.rknn文件

### 3、模型推理

**基于pt模型推理**

```python
python detect.py --weights weights/traffic_500.pt --source video.mp4
```

**基于rknn模型推理图片**

在traffic_detect_image.py修改图片地址和模型地址

```python
ONNX_MODEL = './weights/traffic_500_640x640.onnx'
RKNN_MODEL = './weights/traffic_500.rknn'
IMG_PATH = './data/images/002066.jpg'
DATASET = './dataset.txt'
```

执行代码

```python
python traffic_detect_image.py
```

**基于rknn模型推理视频**

如需修改读取的视频，在traffic_detect_video.py的main函数内修改以下内容，默认视频video.mp4（组委会提供的示例视频）位于当前目录下

```python
capture = cv2.VideoCapture('video.mp4')  # 读取视频
```

执行代码

```python
python traffic_detect_video.py
```

**在当前目录下将会生成result.txt。**

注意：当前目录下已经存在result.txt,这是我们根据示例视频推理后生成的结果，如果运行上面的推理视频代码，则会覆盖此result.txt。

至此所有脚本测试完成

