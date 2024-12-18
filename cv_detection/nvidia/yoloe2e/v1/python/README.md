# YOLO Series TensorRT Python/C++ 

## Support
[YOLOv9](https://github.com/WongKinYiu/yolov9)、[YOLOv8](https://v8docs.ultralytics.com/)、[YOLOv7](https://github.com/WongKinYiu/yolov7)、[YOLOv6](https://github.com/meituan/YOLOv6)、 [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)、 [YOLOV5](https://github.com/ultralytics/yolov5)、[YOLOv3](https://github.com/ultralytics/yolov3)

- [ ] YOLOv10
- [x] YOLOv9
- [x] YOLOv8
- [x] YOLOv7
- [x] YOLOv6
- [x] YOLOX
- [x] YOLOv5
- [x] YOLOv3 

##  Prepare TRT Env 
`Install via Python3.8`
```
pip install tensorrt
pip install cuda-python
```

## YOLOv9
### Generate TRT File 
```shell
python export.py  -o yolov9-c.onnx -e yolov9.trt --end2end --v8 -p fp32
```
### Inference 
```shell
python trt.py -e yolov9.trt  -i src/1.jpg -o yolov9-1.jpg --end2end 
```

## Python Demo
<details><summary> <b>Expand</b> </summary>

1. [YOLOv5](##YOLOv5)
2. [YOLOx](##YOLOX)
3. [YOLOv6](##YOLOV6)
4. [YOLOv7](##YOLOv7)
5. [YOLOv8](##YOLOv8)

## YOLOv8

### Install && Download [Weights](https://github.com/ultralytics/assets/)
```shell
pip install ultralytics
```
### Export ONNX
```Python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.fuse()  
model.info(verbose=False)  # Print model information
model.export(format='onnx')  # TODO: 
```
### Generate TRT File 
```shell
python export.py -o yolov8n.onnx -e yolov8n.trt --end2end --v8 --fp32
```
### Inference 
```shell
python trt.py -e yolov8n.trt  -i src/1.jpg -o yolov8n-1.jpg --end2end 
```


## YOLOv5


```python
!git clone https://github.com/ultralytics/yolov5.git
```

```python
!wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
```


```python
!python yolov5/export.py --weights yolov5n.pt --include onnx --simplify --inplace 
```

### include  NMS Plugin


```python
!python export.py -o yolov5n.onnx -e yolov5n.trt --end2end
```


```python
!python trt.py -e yolov5n.trt  -i src/1.jpg -o yolov5n-1.jpg --end2end 
```

###  exclude NMS Plugin


```python
!python export.py -o yolov5n.onnx -e yolov5n.trt 
```


```python
!python trt.py -e yolov5n.trt  -i src/1.jpg -o yolov5n-1.jpg 
```

## YOLOX 


```python
!git clone https://github.com/Megvii-BaseDetection/YOLOX.git
```


```python
!wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```


```python
!cd YOLOX && pip3 install -v -e . --user
```


```python
!cd YOLOX && python tools/export_onnx.py --output-name ../yolox_s.onnx -n yolox-s -c ../yolox_s.pth --decode_in_inference
```

### include  NMS Plugin


```python
!python export.py -o yolox_s.onnx -e yolox_s.trt --end2end
```


```python
!python trt.py -e yolox_s.trt  -i src/1.jpg -o yolox-1.jpg --end2end 
```

###  exclude NMS Plugin


```python
!python export.py -o yolox_s.onnx -e yolox_s.trt 
```


```python
!python trt.py -e yolox_s.trt  -i src/1.jpg -o yolox-1.jpg 
```

## YOLOv6 


```python
!wget https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.onnx
```

### include  NMS Plugin


```python
!python export.py -o yolov6s.onnx -e yolov6s.trt --end2end
```


```python
!python trt.py -e yolov6s.trt  -i src/1.jpg -o yolov6s-1.jpg --end2end
```

###  exclude NMS Plugin


```python
!python export.py -o yolov6s.onnx -e yolov6s.trt 
```


```python
!python trt.py -e yolov6s.trt  -i src/1.jpg -o yolov6s-1.jpg 
```

## YOLOv7


```python
!git clone https://github.com/WongKinYiu/yolov7.git
```


```python
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```


```python
!pip install -r yolov7/requirements.txt
```


```python
!python yolov7/export.py --weights yolov7-tiny.pt --grid  --simplify
```

### include  NMS Plugin


```python
!python export.py -o yolov7-tiny.onnx -e yolov7-tiny.trt --end2end
```


```python
!python trt.py -e yolov7-tiny.trt  -i src/1.jpg -o yolov7-tiny-1.jpg --end2end
```

###  exclude NMS Plugin


```python
!python export.py -o yolov7-tiny.onnx -e yolov7-tiny-norm.trt
```


```python
!python trt.py -e yolov7-tiny-norm.trt  -i src/1.jpg -o yolov7-tiny-norm-1.jpg
```
</details>






