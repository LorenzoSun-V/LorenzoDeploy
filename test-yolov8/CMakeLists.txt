
cmake_minimum_required(VERSION 3.0)

project(test-yolov8)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

# deepmode
set(INFER_INCLUDE ${ROOT_PATH}/yolov8/src/ ) 
set(INFER_LIBPATH ${ROOT_PATH}/yolov8/build/src )
set(INFER_LIBS yolov8infer)

set(UTILS_INCLUDE ${ROOT_PATH}/utils/src/) 
set(UTILS_LIBPATH ${ROOT_PATH}/utils/build/src/ )
set(UTILS_LIBS utils)

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
    set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_dnn)
endif()

# compile flags
SET(CMAKE_CXX_FLAGS "-std=c++17  -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O3")

if(RTMPSTREAM)
    set(VIDEODECODE_INCLUDE ${ROOT_PATH}/videocodec/src/ ) 
    set(VIDEODECODE_LIBPATH ${ROOT_PATH}/videocodec/build/src )
    set(VIDEODECODE_LIBS videocodec)
    include_directories(${VIDEODECODE_INCLUDE})
    link_directories(${VIDEODECODE_LIBPATH})
    ADD_SUBDIRECTORY(test-rtspclient)
endif()

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${INFER_INCLUDE})
include_directories(${OPENCV_INCLUDE})
include_directories(${UTILS_INCLUDE})

# library path
link_directories(${OPENCV_LIBPATH})
link_directories(${INFER_LIBPATH})
link_directories(${UTILS_LIBPATH})

#subdirectory
ADD_SUBDIRECTORY(test-precious)
ADD_SUBDIRECTORY(test-batchinfer)
ADD_SUBDIRECTORY(test-infer)

