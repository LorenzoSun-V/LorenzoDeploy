#! /bin/bash
###
 # @FilePath: /jack/bt_alg_api/test-fastinfer/build.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-12-23 09:06:06
### 

rm -rf build
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=/home/sysadmin/jack/fastdeploy-linux-x64-gpu-1.0.7
make -j$(nproc)
