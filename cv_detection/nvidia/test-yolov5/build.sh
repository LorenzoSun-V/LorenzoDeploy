#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-06-21 13:09:55
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-12-23 09:06:00
 # @Description: 
### 
# rm -rf build
mkdir build
cd build
cmake .. -DRTMPSTREAM=OFF
make -j$(nproc)
