
cmake_minimum_required(VERSION 3.0)

project(modetest)
#option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# cuda
set(CUDA_INCLUDE /usr/local/cuda/include)
set(CUDA_LIBPATH /usr/local/cuda/lib64)
set(CUDA_LIBS cudart )

set(INFER_INCLUDE ${TRT_PATH}/include)
set(INFER_LIBPATH ${TRT_PATH}/lib)
set(INFER_LIBS nvinfer nvinfer_plugin  nvonnxparser)


# compile flags
SET(CMAKE_CXX_FLAGS "-std=c++11  -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O3")

# include path
include_directories(${INFER_INCLUDE})
include_directories(${CUDA_INCLUDE})

# library path
link_directories(${INFER_LIBPATH})
link_directories(${CUDA_LIBPATH})
#subdirectory
ADD_SUBDIRECTORY(src)
