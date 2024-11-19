<!--
 * @FilePath: /jack/bt_alg_api/test-fastinfer/readme.md
 * @Copyright: 无锡宝通智能科技股份有限公司
 * @Author: jiajunjie@boton-tech.com
 * @LastEditTime: 2024-10-08 15:01:28
-->
## 设置环境变量
```
export LD_LIBRARY_PATH=/home/sysadmin/jack/fastdeploy-linux-x64-gpu-1.0.7/libs
```

## 测试单batch推理
```
./test-infer /home/jack/Downloads/val_imgs/images/192.168.30.55_03_202402040825449_221.jpg /home/jack/Downloads/test/model8_b16m_20240627_cls2_kjg_v0.2.1.onnx 

./test-infer /home/jack/Downloads/val_imgs/images/192.168.30.55_03_202402040825449_221.jpg  /home/jack/Downloads/test/model8_b16m_20240627_cls2_kjg_v0.2.1.onnx  /home/jack/data1/project/bt_alg_api/test-fastinfer/build/test-fastinfer/model_fp16.trt
```
## 测试单batch推理
./test-infer /home/sysadmin/jack/data/XYK20240924/ /home/sysadmin/jack/xray/model5_b64m_20240920_cls10_xray-sub_v0.1.onnx result

## 测试多batch推理
```
./test-batch /home/jack/data2/test/street/imglist.txt /home/jack/data2/test/model2.onnx
```

## 验证模型精度
```
./test-precious  /home/jack/Downloads/val_imgs/images/ /home/jack/Downloads/test/model8_b16m_20240627_cls2_kjg_v0.2.1.onnx  /home/jack/data1/project/bt_alg_api/test-fastinfer/build/test-fastinfer/model_fp16.trt
```