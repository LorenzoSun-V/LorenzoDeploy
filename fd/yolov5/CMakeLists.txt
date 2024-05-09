PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# Specify the fastdeploy library path after downloading and decompression
#option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

message(STATUS "FASTDEPLOY_INSTALL_DIR: " ${FASTDEPLOY_INSTALL_DIR})
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# Add FastDeploy dependent header files
include_directories(${FASTDEPLOY_INCS})

message(STATUS "PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})

# include
include_directories(${PROJECT_SOURCE_DIR}/../include/)
# src
include_directories(${PROJECT_SOURCE_DIR}/../src/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/../src/*)

SET(CMAKE_CXX_FLAGS "-w -W -Wall -Wno-sign-compare -std=c++17 -O3")


# add_executable(detect ${PROJECT_SOURCE_DIR}/detect.cc)
# target_link_libraries(detect ${FASTDEPLOY_LIBS} yaml-cpp)

# add_executable(batch_detect ${PROJECT_SOURCE_DIR}/batch_detect.cc)
# target_link_libraries(batch_detect ${FASTDEPLOY_LIBS})

add_executable(main ${PROJECT_SOURCE_DIR}/main.cc ${SRCS})
target_link_libraries(main ${FASTDEPLOY_LIBS} yaml-cpp)