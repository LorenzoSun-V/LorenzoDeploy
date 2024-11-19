#! /bin/bash

rm -rf build
mkdir build
cd build
cmake ..  -DWITH_E2E_V2=ON
make  -j$(nproc)
