# 1. Install environment

## 1.1 Install OpenCV from source.
1. Download OpenCV source code from git.
    ```
    git clone https://github.com/opencv/opencv.git
    git clone https://github.com/opencv/opencv_contrib.git
    ```

2. Install necessary dependencies.
    ```
    sudo apt-get update
    sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
    sudo apt-get install python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
    ```

3. Build and install OpenCV.
    ```
    cd opencv
    mkdir build && cd build
    cmake -D CMAKE_BUILD_TYPE=Release 
          -D CMAKE_INSTALL_PREFIX=/usr/local 
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
    ```
    Use `cmake -DCMAKE_INSTALL_PREFIX=/path/to/installation ..` if intend to install OpenCV in other locations.

    ```
    make -j12
    make install
    sudo ldconfig
    ```

## 1.2 Install TensorRT
1. Download TensorRT sdk from [Nvidia](https://developer.nvidia.com/nvidia-tensorrt-8x-download) (Usually choose General Availability(GA) rather than Early Access(EA)).

2. 
    ```
    tar -zxvf ${TensorRT-xxx.tar.gz}
    ```

3. 
    ```
    vim ~/.bashrc
    export LD_LIBRARY_PATH=/Path/to/TensorRT-xxx/lib:$LD_LIBRARY_PATH
    source ~/.bashrc
    ```

## 1.3 Install FastDeploy from source.
```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
         -DENABLE_PADDLE_BACKEND=ON \
         -DENABLE_OPENVINO_BACKEND=ON \
         -DENABLE_TRT_BACKEND=ON \
         -DWITH_GPU=ON \
         -DTRT_DIRECTORY=${Path/to/TensorRT} \
         -DCUDA_DIRECTORY=/usr/local/cuda \
         -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk \
         -DENABLE_VISION=ON \
         -DOPENCV_DIRECTORY=${/usr/local/lib/cmake/opencv4}
make -j12
make install
```
The default OpenCV install dir is `/usr/local/lib/cmake/opencv4`, and FastDeploy is installed in ${PWD}/compiled_fastdeploy_sdk.


# FAQ
- Encounter `error while loading shared libraries: libxxx: cannot open shared object file: No such file or directory`.
    
    Use `copy_so.sh` to copy libs into one directory, and then `export LD_LIBRARY_PATH=/path/to/libs` before run.
