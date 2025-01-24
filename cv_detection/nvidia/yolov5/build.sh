#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-07-02 14:40:31
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-12-23 09:05:25
 # @Description: 
### 
# rm -rf build
mkdir build
cd build
cmake ..  -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j$(nproc)
