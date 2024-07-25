#! /bin/bash
rm -rf build
mkdir build
cd build
cmake ..  -DWITH_RK3399PRO_2004=ON 
make -j4
