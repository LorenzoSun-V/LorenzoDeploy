#! /bin/bash
###
 # @FilePath: /bt_alg_api/featextractor/build.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-12-10 10:12:40
### 

rm -rf build
mkdir build
cd build
cmake ..   -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j4
