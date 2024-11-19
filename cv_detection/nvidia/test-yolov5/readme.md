## 设置环境变量
```
export LD_LIBRARY_PATH=/home/mic-710aix/nvlibs
```
## 测试单batch推理
```
./test-infer /home/mic-710aix/model/yolov5s-v7.0-rockchip_20231226_cls3_kjg-jg_v0.1.engine /home/mic-710aix/valimage/frame_1693987947744_35.jpg 
```
## 测试多batch推理
```
./test-batchinfer /home/mic-710aix/model/yolov5s-v7.0-rockchip_20231226_cls3_kjg-jg_v0.1_batch4.engine /home/mic-710aix/valimage/ 
```
## 验证模型精度
```
./test-precious  /home/mic-710aix/model/yolov5s-v7.0-rockchip_20231226_cls3_kjg-jg_v0.1.engine /home/mic-710aix/valimage/ 
```