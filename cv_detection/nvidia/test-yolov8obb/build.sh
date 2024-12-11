#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-07-16 16:10:30
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2024-11-28 17:09:15
 # @Description: 
### 
rm -rf build
mkdir build
cd build
cmake ..  
make -j4
