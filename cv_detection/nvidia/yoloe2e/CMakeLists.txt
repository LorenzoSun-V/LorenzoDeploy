cmake_minimum_required(VERSION 3.0)

if(WITH_E2E_V2)
    project(yoloe2ev2infer)
elseif(WITH_E2E_V1)   
    project(yoloe2ev1infer)
endif()

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../)

#common 
set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

#cuda
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# opencv
MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}") 
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst165_nv_aarch64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst165_nv_aarch64/lib/ )
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "opencv linux") 
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.9.0_x86_x64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.9.0_x86_x64/lib/ ) 
endif()
set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)
MESSAGE(${OPENCV_LIBPATH})
# cuda
set(CUDA_INCLUDE /usr/local/cuda/include)
set(CUDA_LIBPATH /usr/local/cuda/lib64)
set(CUDA_LIBS cudart)

# trt 
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(TRT_INCLUDE ${TRT_PATH}/include)
    set(TRT_LIBPATH ${TRT_PATH}/lib)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")    
    set(TRT_INCLUDE /usr/include/aarch64-linux-gnu/)
    set(TRT_LIBPATH /usr/lib/aarch64-linux-gnu/)
endif()
set(TRT_LIBS nvinfer nvonnxparser nvinfer_plugin)

# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++11 -O3")

# source 
if(WITH_E2E_V2)
    set(UTILS_PATH ${PROJECT_SOURCE_DIR}/v2/)
    file(GLOB_RECURSE SRCS ${UTILS_PATH}/utils/*.cpp)
    file(GLOB LIB_SRCS ${UTILS_PATH}/src/*.cpp)
elseif(WITH_E2E_V1)   
    set(UTILS_PATH ${PROJECT_SOURCE_DIR}/v1/)
    file(GLOB_RECURSE SRCS ${UTILS_PATH}/utils/*.cpp)
    file(GLOB LIB_SRCS ${UTILS_PATH}/src/*.cpp)
endif()

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${OPENCV_INCLUDE})
include_directories(${CUDA_INCLUDE})
include_directories(${TRT_INCLUDE})
include_directories(${UTILS_PATH}/utils)

# library path
link_directories(${OPENCV_LIBPATH})
link_directories(${CUDA_LIBPATH})
link_directories(${TRT_LIBPATH})

#subdirectory
if(WITH_E2E_V2)
  ADD_SUBDIRECTORY(v2/src)
elseif(WITH_E2E_V1) 
  ADD_SUBDIRECTORY(v1/src)
endif()
