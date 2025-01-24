## 设置环境变量
```
export LD_LIBRARY_PATH=/home/mic-710aix/nvlibs
```
## 测试单batch推理
```
./test-infer /home/mic-710aix/model/yolov8m_20240320_cls4_zs_v0.1.engine /home/mic-710aix/valimage/
```
## 测试多batch推理
```
./test-batchinfer /home/mic-710aix/yolov8m_20240320_cls4_zs_v0.1.engine /home/mic-710aix/valimage/ 
```
## 验证模型精度
```
./test-precious  /home/mic-710aix/model/yolov8m_20240320_cls4_zs_v0.1.engine /home/mic-710aix/valimage/ 
```
## 验证RTSP流都中
```
./test-rtspclient /home/mic-710aix/model/yolov8m_20240320_cls4_zs_v0.1.engine rtsp://admin:bts808080@192.168.30.64:554/h264/ch1/main/av_stream 
```