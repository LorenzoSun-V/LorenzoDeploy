<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-11-07 10:43:03
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-11-07 14:50:31
 * @Description: 
-->
## 英伟达目标检测

### 1. fastinfer和test-fastinfer

基于[fastdeploy1.0.7](https://github.com/PaddlePaddle/FastDeploy)接口库开发，支持X86与aarch64平台，支持多种国产硬件，支持yolo5~yolov8多种视觉检测模型，图像分割，OCR等多种场景，全家桶工具仓库。测试代码支持精度验证、性能测试，单图和多图推理。

### 2. onnx2engine
用于将onnx格式模型转为TensorRT格式的模型

### 3. yolov5和test-yolov5
基于[tensorrtx](https://github.com/wang-xinyu/tensorrtx)库开发的视觉检测模型部署代码，支持仿射变换、CUDA前处理，运行前需要将模型转为wts格式后再转为engine格式，其中算子发生改变后对应的量化代码也要改变，特别是对于改过结构的网络不是很友好。测试代码支持精度验证、单图和多图推理。当前yolov5部署代码支持图像分类，图像检测，图像分割场景。

### 4. yolov8和test-yolov8
基于[tensorrtx](https://github.com/wang-xinyu/tensorrtx)库开发的视觉检测模型部署代码，支持仿射变换、CUDA前处理，运行前需要将模型转为wts格式后再转为engine格式，其中算子发生改变后对应的量化代码也要改变，特别是对于改过结构的网络不是很友好。测试代码支持精度验证、单图和多图推理。当前yolov8部署代码支持图像分类，图像检测，旋转目标检测，肢体检测，图像分割场景。

### 5. yolov10和test-yolov10
基于清华的[yolov10](https://github.com/THU-MIG/yolov10)模型部署代码，训练后转成对应onnx，再把onnx转为对应运行格式，目前支持使用openvino进行CPU推理和使用tensorrt进行GPU推理，GPU版本对前处理进行了cuda加速。测试代码支持精度验证，单图和多图推理。

### 6. yoloe2e和test-yoloe2e
设计本代码的目标是统一yolo系列目标检测部署代码，所有yolo模型推理部分使用相同部署代码，减少开发工作量，本仓库分为两个版本，v1版本需要预装python环境，python环境中cuda,tensorrt版本需要和c++中对齐，不然运行会出错，相对友好性较差，v2版本对此做了升级，训练导出onnx后，使用onnx_graphsurgeon对onnx模型进行切片和添加nms操作，分为anchor base版本和anchor free版本，支持x86和arch64平台的英伟达设备运行。

## 瑞星微目标检测

### 1. yolov5与test-yolov5
本代码参考[RK3399Pro_npu](https://github.com/airockchip/RK3399Pro_npu/tree/main/rknn-api/examples/c_demos/rknn_yolov5_demo)项目编写，在RK3399pro嵌入式设备上运行，去除rag入口参数改为opencv读图进行模型推理。