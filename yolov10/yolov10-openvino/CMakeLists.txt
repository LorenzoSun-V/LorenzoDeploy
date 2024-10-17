cmake_minimum_required(VERSION 3.0)

project(yolov10infer-openvino)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)

#common 
set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

# opencv
MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}") 
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst165_nv_aarch64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst165_nv_aarch64/lib/ )
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.9_x86_x64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.9_x86_x64/lib/ ) 
endif()
set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)

# openvino
set(OPENVINO_INCLUDE /opt/intel/openvino/runtime/include)
set(OPENVINO_LIBPATH /opt/intel/openvino/runtime/lib/intel64)
set(OPENVINO_LIBS openvino)

# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++11 -O3")

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${OPENCV_INCLUDE})
include_directories(${OPENVINO_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/utils/)

# library path
link_directories(${OPENCV_LIBPATH})
link_directories(${OPENVINO_LIBPATH})

# source 
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/utils/*.cpp)

#subdirectory
ADD_SUBDIRECTORY(src)
