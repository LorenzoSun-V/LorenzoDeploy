#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-07-16 16:10:30
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-10-17 11:57:16
 # @Description: 
### 
rm -rf build
mkdir build
cd build
cmake ..  
make -j$(nproc)
