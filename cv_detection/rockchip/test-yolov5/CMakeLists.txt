
cmake_minimum_required(VERSION 3.0)

project(modetest)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../)

set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

# infer
set(RKINFER_INCLUDE ${ROOT_PATH}/yolov5/src/ ) 
set(RKINFER_LIBPATH ${ROOT_PATH}/yolov5/build/src )
set(RKINFER_LIBS yolov5)


set(UTILS_INCLUDE ${ROOT_PATH}/utils/src/) 
set(UTILS_LIBPATH ${ROOT_PATH}/utils/build/src/ )
set(UTILS_LIBS utils)
message(${UTILS_LIBPATH})

# opencv
set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/include/ )
set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/lib/  ) 
set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)

if(TESTRTSP)
    # videostream
    set(VIDEOSTREAM_INCLUDE ${ROOT_PATH}/videocodec/gstreamer/src) 
    set(VIDEOSTREAM_LIBPATH ${ROOT_PATH}/videocodec/gstreamer/build/src )
    set(VIDEOSTREAM_LIBS videocodec)
    include_directories(${VIDEOSTREAM_INCLUDE})
    link_directories(${VIDEOSTREAM_LIBPATH})
    ADD_SUBDIRECTORY(test-rtsp)
endif()

# compile flags
SET(CMAKE_CXX_FLAGS "-std=c++11  -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -O3")

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${RKINFER_INCLUDE})
include_directories(${OPENCV_INCLUDE})
include_directories(${UTILS_INCLUDE})

# library path
link_directories(${OPENCV_LIBPATH})
link_directories(${RKINFER_LIBPATH})
link_directories(${UTILS_LIBPATH})

#subdirectory
ADD_SUBDIRECTORY(test-infer)
ADD_SUBDIRECTORY(test-precious)
ADD_SUBDIRECTORY(test-video)
