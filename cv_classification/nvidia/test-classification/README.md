<!--
 * @Author: BTZN0325 sunjiahui@boton-tech.com
 * @Date: 2024-10-28 10:58:00
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2024-12-16 17:37:05
 * @Description: 
-->
## 1. 配置环境变量

```
export LD_LIBRARY_PATH=/home/sysadmin/jack/TensorRT-8.5.2.2/lib
```

## 2. 构建并安装 test-classification

```
    cd test-classification
    sh build.sh
```

* 测试单batch 

```
    cd build/test-infer
    ./test-infer ${image_folder} ${model_path}
```

* image_folder: 待推理的图片文件夹路径
* model_path: tensorrt模型路径

example:
```
 ./test-infer /home/sysadmin/jack/yolo/ultralytics/runs/xray_joint/model11_b16n-cls512_20241213_cls32_xray_gs_v0.4/weights/best.bin /data/bt/cls/xray-gs/joint_v0.4/trainval/val/10
 
  /data/bt/cls/xray-gs/LabeledData/xinyuankuang/XYK20241209/ 
```