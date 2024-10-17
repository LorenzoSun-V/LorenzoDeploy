#! /bin/bash

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/home/ubuntu/TensorRT-8.5.2.2/ -DWITH_E2E_V2=ON
make  -j$(nproc)
