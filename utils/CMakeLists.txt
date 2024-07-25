cmake_minimum_required(VERSION 3.0)

project(utils)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../)

#common 
set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

# opencv
if(WITH_NVIDIA_2004)
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst165_nv_aarch64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst165_nv_aarch64/lib/ ) 
    set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)
elseif(WITH_NVIDIA_1804)
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst145_nv_aarch64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst145_nv_aarch64/lib/ ) 
    set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)
elseif(WITH_RK3399PRO_2004)
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst162_rk_aarch64/lib/  ) 
    set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)
endif()

# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++11 -O3")

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${OPENCV_INCLUDE})

# library path
link_directories(${OPENCV_LIBPATH})


#subdirectory
ADD_SUBDIRECTORY(src)
