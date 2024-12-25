cmake_minimum_required(VERSION 3.0)

project(yolodnninfer)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../)

#common 
set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

# opencv
MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}") 
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst165_nv_aarch64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8.0_gst165_nv_aarch64/lib/ )
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(OPENCV_INCLUDE ${ROOT_PATH}/3rdlibs/opencv4.8_x86_64/include/ )
    set(OPENCV_LIBPATH ${ROOT_PATH}/3rdlibs/opencv4.8_x86_64/lib/ )
    # set(OPENCV_INCLUDE /home/sysadmin/lorenzo/env/opencv4.8_x86_64/include/opencv4 )
    # set(OPENCV_LIBPATH /home/sysadmin/lorenzo/env/opencv4.8_x86_64/lib/ )
endif()
set(OPENCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_dnn)


# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++17 -O3 -msse4.2 -mavx512f")

# source
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/utils/*.cpp)
file(GLOB LIB_SRCS ${UTILS_PATH}/src/*.cpp)

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${OPENCV_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/utils)

# library path
link_directories(${OPENCV_LIBPATH})

#subdirectory
ADD_SUBDIRECTORY(src)