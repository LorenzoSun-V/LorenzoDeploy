#! /bin/bash
###
 # @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yolov10/yolov10-trt/build.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-11-19 14:37:36
### 

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/home/sysadmin/jack/TensorRT-8.5.2.2/lib
make -j4
