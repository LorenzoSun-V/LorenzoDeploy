#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-06-28 11:31:35
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-07-01 10:35:53
 # @Description: 
### 

mkdir build
cd build
cmake .. -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j4