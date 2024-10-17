<!--
 * @FilePath: /jack/bt_alg_api/test-yoloe2e/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-09-30 15:17:31
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
## 验证模型精度
```
./test-precious  /home/mic-710aix/bt_zs_4x_api/yolov10/build/onnx2engine/model10_b16m_20240619_cls2_kjg_v0.1.10.engine /home/mic-710aix/valimage/ 
```