<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2025-01-06 10:04:40
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2025-01-06 10:06:52
 * @Description: 
-->

## 英伟达分割

### 1. yoloseg和test-yoloseg

基于[tensorrtx](https://github.com/wang-xinyu/tensorrtx)库开发的通用YOLO实例分割代码。运行前需要将pt模型导出为onnx格式再转为engine，同时YOLOv8和YOLOv11等Detect Head的输出维度需要permute，从[1, 8400, n] -> [1, n, 8400]，和YOLOv5对齐。测试代码支持单图和多图推理。