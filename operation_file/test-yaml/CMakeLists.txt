cmake_minimum_required(VERSION 3.0)

project(testyaml)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../..)
message(${ROOT_PATH})

#common 
set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

# yaml
set(YAML_INCLUDE ${ROOT_PATH}/operation_file/yaml/src/ )
set(YAML_LIBPATH ${ROOT_PATH}/operation_file/yaml/build/src) 
set(YAML_LIBS yaml)

# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++11 -O3")

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${YAML_INCLUDE})

# library path
link_directories(${YAML_LIBPATH})

#subdirectory
ADD_SUBDIRECTORY(test)
