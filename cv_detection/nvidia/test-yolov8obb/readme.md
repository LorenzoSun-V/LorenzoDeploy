<!--
 * @FilePath: /jack/github/bt_alg_api/cv_detection/nvidia/test-yolov8obb/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-12-10 16:47:02
-->
## 设置环境变量
```
export LD_LIBRARY_PATH=/home/sysadmin/jack/nvlibs/
```
## 测试单batch推理
```
./test-infer /home/sysadmin/jack/models/model8_b32s-obb_20241203_cls2_hw_obb_v0.1_b4.bin /home/sysadmin/jack/zs_obb_test     
```
## 测试多batch推理
```
./test-batchinfer /home/sysadmin/gmz/yolo/ultralytics_obb/runs/DOTA/model11_b32s-obb_20241205_cls2_hw_obb_v0.1/weights/model11_b32s-obb_20241205_cls2_hw_obb_v0.1_b4.bin /home/sysadmin/jack/zs_obb_test
```
## 验证模型精度
```
./test-precious  /home/sysadmin/jack/models/model8_b32s-obb_20241203_cls2_hw_obb_v0.1_b4.bin /home/sysadmin/jack/zs_obb_val
```