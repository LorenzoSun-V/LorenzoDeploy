#! /bin/bash

mkdir build
cd build
cmake ..  -DWITH_x86_2004=ON -DTRT_PATH=/lorenzo/env/install/TensorRT-8.6.1.6
make -j4
