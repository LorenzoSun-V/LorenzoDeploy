#! /bin/bash
###
 # @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yoloe2e/build.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-12-23 15:54:33
### 

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6 -DWITH_E2E_V2=ON
make  -j$(nproc)
