#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-06-21 13:09:55
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-12-23 09:01:23
 # @Description: 
### 
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)