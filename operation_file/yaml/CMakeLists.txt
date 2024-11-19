cmake_minimum_required(VERSION 3.0)

project(yaml)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)

#common 
set(COMMON_INCLUDE  ${ROOT_PATH}/include/ )

# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++11 -O3")
#cmake_print_variable(COMMON_INCLUDE)
# include path
include_directories(${COMMON_INCLUDE})
# yaml-cpp
MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}") 
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(YAML_INCLUDE ${ROOT_PATH}/3rdlibs/yaml-cpp/aarch64/include/ )
    set(YAML_LIBPATH ${ROOT_PATH}/3rdlibs/yaml-cpp/aarch64/lib/ )
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(YAML_INCLUDE ${ROOT_PATH}/3rdlibs/yaml-cpp/x86_x64/include/ )
    set(YAML_LIBPATH ${ROOT_PATH}/3rdlibs/yaml-cpp/x86_x64/lib/ ) 
endif()
include_directories(YAML_INCLUDE)
link_directories(YAML_LIBPATH)
set(YAML_LIBS yaml-cpp)


#subdirectory
ADD_SUBDIRECTORY(src)
