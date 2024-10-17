#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-06-28 08:29:19
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-07-25 14:18:57
 # @Description: 
### 
rm -rf build
mkdir build
cd build
cmake ..  -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j4
