cmake_minimum_required(VERSION 3.0)

project(modetest)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

# fastinfer
set(FASTINFER_INCLUDE ${ROOT_PATH}/cv_detection/nvidia/fastinfer/src/) 
set(FASTINFER_LIBPATH ${ROOT_PATH}/cv_detection/nvidia/fastinfer/build/src )
set(FASTINFER_LIBS fastinfer)

# opencv
set(OPENCV_INCLUDE ${FASTDEPLOY_INSTALL_DIR}/third_libs/install/opencv/include/ )
set(OPENCV_LIBPATH ${FASTDEPLOY_INSTALL_DIR}/third_libs/install/opencv/lib64/) 
set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)

# fastdeploy
set(FASTDEPLOY_LIBPATH ${FASTDEPLOY_INSTALL_DIR}/libs/) 
set(FASTDEPLOY_LIBS fastdeploy)

# compile flags
SET(CMAKE_CXX_FLAGS "-std=c++17  -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O3")

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${FASTINFER_INCLUDE})
include_directories(${OPENCV_INCLUDE})

# library path
link_directories(${OPENCV_LIBPATH})
link_directories(${FASTINFER_LIBPATH})
link_directories(${FASTDEPLOY_LIBPATH})

#subdirectory
ADD_SUBDIRECTORY(test-infer)
ADD_SUBDIRECTORY(test-fps)
ADD_SUBDIRECTORY(test-batchinfer)
ADD_SUBDIRECTORY(test-precious)
