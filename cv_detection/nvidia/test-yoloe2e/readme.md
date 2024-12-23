<!--
 * @FilePath: /jack/github/bt_alg_api/cv_detection/nvidia/test-yoloe2e/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-06 14:41:53
-->
## 设置环境变量
```
export LD_LIBRARY_PATH=/home/sysadmin/jack/nvlibs
```
## 测试单batch推理
```
./test-infer  /home/sysadmin/jack/yolo/ultralytics/yolo11m_1_nms.bin /home/sysadmin/jack/yolo/ultralytics/images
```
## 测试多batch推理
```
./test-batchinfer /home/sysadmin/jack/models/model8e2e_b16m_20240627_cls2_kjg_v0.2.1_b4_jack.bin /home/sysadmin/jack/images/zs/
```
## 验证模型精度
```
./test-precious /home/sysadmin/jack/ultralytics/runs/detect/v8s_hw_cls2_320_v0.17_1104/weights/model8e2e_b16s_20241104_cls2_320_v0.17.bin /data/bt/hw_multi/cls2_hw_v0.1.7.2/val_img/images/ 
```
