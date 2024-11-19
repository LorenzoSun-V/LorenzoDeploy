cmake_minimum_required(VERSION 3.10)
project(SearchFeature)

set(CMAKE_CXX_STANDARD 17)  # 为了支持filesystem
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置Eigen库的路径，这里假设Eigen库位于系统特定路径下
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")


add_executable(SearchFeature test.cpp)
target_include_directories(SearchFeature PRIVATE ${EIGEN3_INCLUDE_DIR})

find_package(Eigen3 REQUIRED)
target_link_libraries(SearchFeature Eigen3::Eigen)
