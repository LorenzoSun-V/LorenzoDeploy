cmake_minimum_required(VERSION 3.0)

project(featextractor)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../)

#common 
set(COMMON_INCLUDE ${ROOT_PATH}/include/ )
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)

# opencv
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/lib/ ) 
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "opencv linux") 
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8_x86_64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8_x86_64/lib/ ) 
endif()
set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)
# cuda
set(CUDA_INCLUDE /usr/local/cuda/include)
set(CUDA_LIBPATH /usr/local/cuda/lib64)
set(CUDA_LIBS cudart)
# trt 
set(TRT_INCLUDE ${TRT_PATH}/include)
set(TRT_LIBPATH ${TRT_PATH}/lib)
set(TRT_LIBS nvinfer nvonnxparser)

# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++11 -O3")

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${OPENCV_INCLUDE})
include_directories(${CUDA_INCLUDE})
include_directories(${TRT_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/utils/)

# library path
link_directories(${OPENCV_LIBPATH})
link_directories(${CUDA_LIBPATH})
link_directories(${TRT_LIBPATH})

# source 
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/utils/*.cpp)

#subdirectory
ADD_SUBDIRECTORY(src)