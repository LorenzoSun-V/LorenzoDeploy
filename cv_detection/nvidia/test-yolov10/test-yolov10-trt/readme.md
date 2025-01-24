<!--
 * @FilePath: /jack/bt_alg_api/cv_detection/nvidia/test-yolov10/test-yolov10-trt/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-11-19 15:29:52
-->
## 设置环境变量
```
export LD_LIBRARY_PATH=/home/mic-710aix/nvlibs
```
## 测试单batch推理
```
./test-infer /home/mic-710aix/model/model10_b16m_20240701_cls2_kjg_v0.2.1.bin /home/mic-710aix/valimage/
```
## 验证模型精度
```
./test-precious  /home/mic-710aix/bt_zs_4x_api/yolov10/build/onnx2engine/model10_b16m_20240619_cls2_kjg_v0.1.10.engine /home/mic-710aix/valimage/ 
```