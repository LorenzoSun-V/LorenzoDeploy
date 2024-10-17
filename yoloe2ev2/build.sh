#! /bin/bash

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/home/ubuntu/TensorRT-8.5.2.2/
make  -j$(nproc)
