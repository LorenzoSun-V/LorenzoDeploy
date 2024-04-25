PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

message(STATUS "PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
# fastdeploy
message(STATUS "FASTDEPLOY_INSTALL_DIR: " ${FASTDEPLOY_INSTALL_DIR})
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)
include_directories(${FASTDEPLOY_INCS})

# include
include_directories(/lorenzo/deploy/LorenzoDeploy/fd/include)

SET(CMAKE_CXX_FLAGS "-w -W -Wall -Wno-sign-compare -std=c++17 -O3")


add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer.cc)
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})

add_executable(multi_infer_demo ${PROJECT_SOURCE_DIR}/multi_infer.cc)
target_link_libraries(multi_infer_demo ${FASTDEPLOY_LIBS})

add_executable(detect ${PROJECT_SOURCE_DIR}/detect.cc)
target_link_libraries(detect ${FASTDEPLOY_LIBS} yaml-cpp)
