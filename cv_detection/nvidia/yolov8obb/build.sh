#! /bin/bash
###
 # @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yolov10/yolov10-trt/build.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-12-12 13:27:47
### 

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j$(nproc)
