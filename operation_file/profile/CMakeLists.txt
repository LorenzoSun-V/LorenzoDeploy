cmake_minimum_required(VERSION 3.0)

project(profile)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)

#common 
set(COMMON_INCLUDE  ${ROOT_PATH}/include/ )

# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++11 -O3")
#cmake_print_variable(COMMON_INCLUDE)
# include path
include_directories(${COMMON_INCLUDE})

#subdirectory
ADD_SUBDIRECTORY(src)
