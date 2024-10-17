<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-10-17 14:21:19
 * @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 * @LastEditTime: 2024-10-17 14:21:22
 * @Description: 
-->
## 设置环境变量
```
export LD_LIBRARY_PATH=/home/mic-710aix/nvlibs
export LD_LIBRARY_PATH=/home/ubuntu/nv-libs
```
## 测试单batch推理
```
./test-infer  /home/jack/TensorRT-8.5.2.2/bin/yolov9-m-converted_1_nms.engine /home/jack/Downloads/test
```
## 验证模型精度
```
./test-precious  /home/mic-710aix/bt_zs_4x_api/yolov10/build/onnx2engine/model10_b16m_20240619_cls2_kjg_v0.1.10.engine /home/mic-710aix/valimage/ 
```