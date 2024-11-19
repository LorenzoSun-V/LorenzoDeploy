#! /bin/bash

rm -rf build
mkdir build
cd build
cmake .. -DTESTRTSP=OFF
make 
