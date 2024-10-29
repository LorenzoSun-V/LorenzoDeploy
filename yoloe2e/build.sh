
trt_path=/lorenzo/env/install/TensorRT-8.6.1.6/

rm -rf build
mkdir build
cd build
cmake .. -DTRT_PATH=${trt_path} -DWITH_E2E_V2=ON
make  -j$(nproc)
