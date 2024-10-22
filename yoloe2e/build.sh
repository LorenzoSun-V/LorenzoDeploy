#! /bin/bash
###
 # @Author: BTZN0325 sunjiahui@boton-tech.com
 # @Date: 2024-10-17 15:00:03
 # @LastEditors: BTZN0325 sunjiahui@boton-tech.com
 # @LastEditTime: 2024-10-21 08:55:04
 # @Description: 
### 
trt_path=/lorenzo/env/install/TensorRT-8.6.1.6/

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=${trt_path} -DWITH_E2E_V2=ON
make  -j$(nproc)
