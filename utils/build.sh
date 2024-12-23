#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-11-19 16:46:52
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-12-12 14:29:32
 # @Description: 
### 
rm -rf build
mkdir build
cd build
cmake ..  
make -j$(nproc)
