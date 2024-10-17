#! /bin/bash
###
 # @FilePath: /bt_alg_api/yolov10/yolov10-trt/build.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-10-17 13:09:44
### 

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j4
