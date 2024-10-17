#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-07-26 14:07:49
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-10-17 13:09:24
 # @Description: 
### 

rm -rf build
mkdir build
cd build
cmake ..
make -j4
