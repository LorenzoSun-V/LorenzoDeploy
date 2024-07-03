#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-07-02 14:40:31
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-07-02 15:19:49
 # @Description: 
### 
# rm -rf build
mkdir build
cd build
cmake .. -DWITH_x86_2004=ON -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j4
