<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-10-17 14:20:00
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-02-20 13:27:14
 * @Description: 
-->
# Deployment of Deep Learning with different frameworks

## YOLO Series End-to-End Detection with TensorRT C++

This project is deployed with TensorRT. Support YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv11, YOLOv12, all varified on 3090 GPU.
Please refer to [YOLOe2e README](cv_detection/nvidia/yoloe2e/v2/python/README.md).


## YOLOv8 OBB with TensorRT C++

Please refer to [YOLOv8OBB README](cv_detection/nvidia/test-yolov8obb/README.md).

The code may be adapted to YOLOv11OBB, but the adaptation is not tested.


## YOLO Series Instance Segmentation with TensorRT C++

Support YOLOv5-seg, YOLOv8-seg, YOLOv11-seg, all varified on 3090 GPU.
Please refer to [YOLOSeg README](cv_segmentation/nvidia/README.md).


## Fastdeploy

This project is deployed with FastDeploy.
Please refer to this [document](fd/README.md).


## tensorrtx

This project is deployed with tensorrtx.