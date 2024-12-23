
cmake_minimum_required(VERSION 3.0)

project(onnx2engine)

# cuda
set(CUDA_INCLUDE /usr/local/cuda/include)
set(CUDA_LIBPATH /usr/local/cuda/lib64)
set(CUDA_LIBS cudart )

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(TRT_INCLUDE ${TRT_PATH}/include)
    set(TRT_LIBPATH ${TRT_PATH}/lib)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")    
    set(TRT_INCLUDE /usr/include/aarch64-linux-gnu/)
    set(TRT_LIBPATH /usr/lib/aarch64-linux-gnu/)
endif()
set(TRT_LIBS nvinfer nvinfer_plugin  nvonnxparser )

# compile flags
SET(CMAKE_CXX_FLAGS "-std=c++11  -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O3")

# include path
include_directories(${TRT_INCLUDE})
include_directories(${CUDA_INCLUDE})

# library path
link_directories(${TRT_LIBPATH})
link_directories(${CUDA_LIBPATH})

#subdirectory
ADD_SUBDIRECTORY(src)
