
cmake_minimum_required(VERSION 3.0)

project(test-featinfer)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

# infer
set(FEATINFER_INCLUDE ${ROOT_PATH}/cv_classification/rockchip/featextractor/src/ ) 
set(FEATINFER_LIBPATH ${ROOT_PATH}/cv_classification/rockchip/featextractor/build/src )
set(FEATINFER_LIBS featinfer)

# opencv
set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/include/ )
set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/lib/  ) 
set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)

# compile flags
SET(CMAKE_CXX_FLAGS "-std=c++11  -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O3")

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${OPENCV_INCLUDE})
include_directories(${FEATINFER_INCLUDE})

# library path
link_directories(${OPENCV_LIBPATH})
link_directories(${FEATINFER_LIBPATH})


#subdirectory
ADD_SUBDIRECTORY(test-infer)
