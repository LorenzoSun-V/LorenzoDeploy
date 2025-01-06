#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-12-25 09:45:42
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-12-25 09:49:33
 # @Description: 
### 
rm -rf build
mkdir build
cd build
cmake ..  
make -j$(nproc)
