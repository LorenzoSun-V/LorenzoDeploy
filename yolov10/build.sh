#! /bin/bash

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j4
