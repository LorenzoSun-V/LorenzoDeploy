cmake_minimum_required(VERSION 3.0)

# source 
file(GLOB LIB_SRCS ./*.cpp)

# shared library 
ADD_LIBRARY(fastinfer SHARED ${LIB_SRCS})
TARGET_LINK_LIBRARIES(fastinfer ${OPENCV_LIBS} ${FASTDEPLOY_LIBS})

