#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-06-28 11:31:35
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-12-23 09:06:12
 # @Description: 
### 
rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/home/sysadmin/jack/TensorRT-8.5.2.2
make -j$(nproc)
