#! /bin/bash
###
 # @FilePath: /jack/github/bt_alg_api/cv_segmentation/nvidia/yolov5seg/build.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-12-27 13:29:16
### 

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j$(nproc)
