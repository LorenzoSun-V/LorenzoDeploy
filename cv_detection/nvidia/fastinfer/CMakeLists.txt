cmake_minimum_required(VERSION 3.0)

project(fastinfer)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

#common 
set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

# include path
include_directories(${COMMON_INCLUDE})

ADD_SUBDIRECTORY(src)
