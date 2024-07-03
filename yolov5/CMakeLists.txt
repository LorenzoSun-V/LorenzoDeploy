cmake_minimum_required(VERSION 3.0)

project(yolov5infer)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../)

#common 
set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# opencv
if(WITH_NVIDIA_2004)
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst145_nv_aarch64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst145_nv_aarch64/lib/ ) 
    set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)
elseif(WITH_ROCKCHIP_2004)
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/lib/ ) 
    set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)
elseif(WITH_x86_2004)
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.9_x86_x64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.9_x86_x64/lib/ ) 
    set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_dnn)  # opencv4.5.2_x86_x64 opencv4.9_x86_x64
endif()

# cuda
set(CUDA_INCLUDE /usr/local/cuda/include)
set(CUDA_LIBPATH /usr/local/cuda/lib64)
set(CUDA_LIBS cudart)
# trt 
set(TRT_INCLUDE ${TRT_PATH}/include)
set(TRT_LIBPATH ${TRT_PATH}/lib)
set(TRT_LIBS nvinfer)
# trt_jetson
# set(TRT_INCLUDE /usr/include/aarch64-linux-gnu/)
# set(TRT_LIBPATH /usr/lib/aarch64-linux-gnu/)

# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++11 -O3")

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${OPENCV_INCLUDE})
include_directories(${CUDA_INCLUDE})
include_directories(${TRT_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/utils/)
#include_directories(${PROJECT_SOURCE_DIR}/plugin/)

# library path
link_directories(${OPENCV_LIBPATH})
link_directories(${CUDA_LIBPATH})
link_directories(${TRT_LIBPATH})

# source 
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/utils/*.cpp ${PROJECT_SOURCE_DIR}/utils/*.cu)

#subdirectory
ADD_SUBDIRECTORY(src)
