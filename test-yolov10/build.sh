#! /bin/bash

mkdir build
cd build
cmake .. -DWITH_x86_2004=ON -DWITH_VIDEO=OFF
make 
