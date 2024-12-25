cmake_minimum_required(VERSION 3.0)

project(featinfer)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

#common 
set(COMMON_INCLUDE ${ROOT_PATH}/include/ )
message(${COMMON_INCLUDE})
set(RKNN_INCLUDE ${ROOT_PATH}/3rdlibs/rknn-api/include )
set(RKNN_LIBPATH ${ROOT_PATH}/3rdlibs/rknn-api/lib64/) 
set(RKNN_LIBS rknn_api )

# opencv
set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/include/ )
set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/lib/  ) 
set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)


# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++11 -O3")

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${OPENCV_INCLUDE})
include_directories(${RKNN_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/utils/)


# library path
link_directories(${OPENCV_LIBPATH})
link_directories(${RKNN_LIBPATH})


#subdirectory
ADD_SUBDIRECTORY(src)
