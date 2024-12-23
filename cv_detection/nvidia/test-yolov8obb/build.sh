#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-07-16 16:10:30
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-12-12 13:27:51
 # @Description: 
### 
rm -rf build
mkdir build
cd build
cmake ..  
make -j$(nproc)
