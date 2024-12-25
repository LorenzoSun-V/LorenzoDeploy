cmake_minimum_required(VERSION 3.0)

# source 
file(GLOB LIB_SRCS ./*.cpp)

# shared library 
ADD_LIBRARY(${CMAKE_PROJECT_NAME} SHARED ${LIB_SRCS} ${SRCS})
TARGET_LINK_LIBRARIES(${CMAKE_PROJECT_NAME} ${OPENCV_LIBS})

