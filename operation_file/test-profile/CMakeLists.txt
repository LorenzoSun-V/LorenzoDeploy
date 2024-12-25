cmake_minimum_required(VERSION 3.0)

project(testprofile)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../..)
message(${ROOT_PATH})

#common 
set(COMMON_INCLUDE ${ROOT_PATH}/include/ )

# profile
set(PROFILE_INCLUDE ${ROOT_PATH}/operation_file/profile/src/ )
set(PROFILE_LIBPATH ${ROOT_PATH}/operation_file/profile/build/src) 
set(PROFILE_LIBS profile )

# compile flags
SET(CMAKE_CXX_FLAGS "-W -Wall -Wno-sign-compare -fPIC -std=c++11 -O3")

# include path
include_directories(${COMMON_INCLUDE})
include_directories(${PROFILE_INCLUDE})

# library path
link_directories(${PROFILE_LIBPATH})

#subdirectory
ADD_SUBDIRECTORY(test)
