#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-06-28 08:29:19
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-11-20 08:42:22
 # @Description: 
### 
rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j$(nproc)
