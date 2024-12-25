#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-02-05 06:45:21
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-02-06 01:50:34
 # @Description: 
### 
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=/lorenzo/deploy/FastDeploy/build/compiled_fastdeploy_sdk
make -j8
# export LD_LIBRARY_PATH=/lorenzo/deploy/FastDeploy/build/compiled_fastdeploy_sdk/libs
