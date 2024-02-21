PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

# Specify the fastdeploy library path after downloading and decompression
#option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

message(STATUS "FASTDEPLOY_INSTALL_DIR: " ${FASTDEPLOY_INSTALL_DIR})
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# Add FastDeploy dependent header files
include_directories(${FASTDEPLOY_INCS})

message(STATUS "PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})

# add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer.cc)
# target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})

# add_executable(multi_infer_demo ${PROJECT_SOURCE_DIR}/multi_infer.cc)
# target_link_libraries(multi_infer_demo ${FASTDEPLOY_LIBS})

add_executable(detect ${PROJECT_SOURCE_DIR}/detect.cc)
target_link_libraries(detect ${FASTDEPLOY_LIBS})
