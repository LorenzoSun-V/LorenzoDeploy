#! /bin/bash
###
 # @FilePath: /bt_alg_api/test-utils/build.sh
 # @Copyright: 无锡宝通智能科技股份有限公司
 # @Author: jiajunjie@boton-tech.com
 # @LastEditTime: 2024-09-05 17:45:50
### 
rm -rf build
mkdir build
cd build
cmake ..  
make 
