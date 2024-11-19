#! /bin/bash
###
 # @FilePath: /jack/bt_alg_api/cv_detection/nvidia/yoloe2e/build.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-11-07 16:09:31
### 

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/home/sysadmin/jack/TensorRT-8.5.2.2 -DWITH_E2E_V2=ON
make  -j$(nproc)
