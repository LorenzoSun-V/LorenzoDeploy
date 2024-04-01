cmake_minimum_required(VERSION 3.10)

project(test)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# OpenCV
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
set(OPENCV_INCLUDE /usr/local/include/opencv4 )
set(OPENCV_LIBPATH /usr/local/lib ) 
set(OpenCV_LIBS opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_video opencv_dnn)
include_directories(${OPENCV_INCLUDE})
link_directories(${OPENCV_LIBPATH})

add_executable(test test.cc ${SRCS})
target_link_libraries(test ${OpenCV_LIBS})

