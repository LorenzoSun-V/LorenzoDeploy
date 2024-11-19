#! /bin/bash
###
 # @FilePath: /bt_alg_api/featextractor/build.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-09-03 17:40:13
### 

mkdir build
cd build
cmake ..   -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j4
