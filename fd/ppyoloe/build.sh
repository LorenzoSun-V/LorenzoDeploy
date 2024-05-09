mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=/lorenzo/deploy/FastDeploy/build/compiled_fastdeploy_sdk
make -j8
# export LD_LIBRARY_PATH=/lorenzo/deploy/FastDeploy/build/compiled_fastdeploy_sdk/libs