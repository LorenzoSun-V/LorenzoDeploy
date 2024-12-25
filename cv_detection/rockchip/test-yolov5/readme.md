## 设置环境变量
```
#将ffmeida rknnrt opencv等用到的库拷贝到rk-libs目录
export LD_LIBRARY_PATH=/home/firefly/rk-libs
```

## 测试图像
```
./test-infer /home/firefly/model/model5-v7.0-rockchip_b16m_20240604_cls2_kjg_v0.1.9_u8.rknn /home/firefly/cls2_ir_v0.1/val_images/frame_1698368211332_75.jpg 
```

## 测试视频
```
./test-video /home/firefly/model/model5-v7.0-rockchip_b16m_20240604_cls2_kjg_v0.1.9_u8.rknn /home/firefly/test.mp4
```

## 测试精度
```
./test-precious /home/firefly/model/model5-v7.0-rockchip_b16m_20240604_cls2_kjg_v0.1.9_u8.rknn /home/firefly/cls2_ir_v0.1/val_images/
``` 

## 测试视频流
```
./test-rtsp  /home/firefly/model/model5-v7.0-rockchip_b16m_20240604_cls2_kjg_v0.1.9_u8.rknn  rtsp://192.168.137.15:554/stream0
``` 