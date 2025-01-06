<!--
 * @FilePath: /jack/github/bt_alg_api/cv_segmentation/nvidia/test-yolov5seg/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2025-01-06 11:31:02
-->
## 设置环境变量
```
export LD_LIBRARY_PATH=3rdlibs/opencv4.8_x86_64/lib
```

## 测试单batch推理
```
./build/test-infer/test-infer /lorenzo/bt_repo/yolov5/yolov5-v7.0/data/images /lorenzo/bt_repo/yolov5/yolov5-v7.0/weights/yolov5m-seg_b1.engine 0
```
参数说明：YOLOv5等anchor-based分割参数为1，YOLOv8/yolov11分割参数为0。

## 测试多batch推理
```
./build/test-batchinfer/test-batchinfer /lorenzo/bt_repo/yolov5/yolov5-v7.0/data/images /lorenzo/bt_repo/yolov5/yolov5-v7.0/weights/yolov5m-seg_b4.engine 0
```
参数说明：YOLOv5等anchor-based分割参数为1，YOLOv8/yolov11分割参数为0。