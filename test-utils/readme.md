## 设置环境变量
```
export LD_LIBRARY_PATH=/home/mic-710aix/nvlibs
export LD_LIBRARY_PATH=/home/ubuntu/nv-libs
```
## 测试单batch推理
```
./test-infer /home/mic-710aix/model/model10_b16m_20240701_cls2_kjg_v0.2.1_e2e.bin /home/mic-710aix/valimage/
```
## 验证模型精度
```
./test-precious  /home/mic-710aix/bt_zs_4x_api/yolov10/build/onnx2engine/model10_b16m_20240619_cls2_kjg_v0.1.10.engine /home/mic-710aix/valimage/ 
```