<!--
 * @FilePath: /jack/bt_alg_api/cv_detection/nvidia/test-yoloe2e/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-11-07 16:34:06
-->
## 设置环境变量
```
export LD_LIBRARY_PATH=/home/mic-710aix/nvlibs
export LD_LIBRARY_PATH=/home/sysadmin/jack/nvlibs
```
## 测试单batch推理
```
./test-infer  /home/sysadmin/jack/yolo/ultralytics/yolo11m_1_nms.bin /home/sysadmin/jack/yolo/ultralytics/images
```
## 测试多batch推理
```
./test-batchinfer  /home/jack/TensorRT-8.5.2.2/bin/yolo11m_1_nms.engine /home/jack/Downloads/test3

```
## 验证模型精度
```
./test-precious /home/sysadmin/jack/ultralytics/runs/detect/v8s_hw_cls2_320_v0.17_1104/weights/model8e2e_b16s_20241104_cls2_320_v0.17.bin /data/bt/hw_multi/cls2_hw_v0.1.7.2/val_img/images/ 
```
